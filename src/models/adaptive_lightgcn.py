"""AdaptiveLightGCN: LightGCN with learnable layer aggregation weights.

Stage B experimental extension:
Replaces the fixed uniform aggregation weights \\alpha_k = 1 / (K + 1) with
trainable parameters \\theta_k \\in \\mathbb{R}:

    \\alpha_k = \\operatorname{softmax}(\\theta)_k = \\frac{\\exp(\\theta_k)}{\\sum_{j=0}^K \\exp(\\theta_j)}

    E^* = \\sum_{k=0}^K \\alpha_k E^{(k)}

Initialization:
    \\theta_k = 0 for all k \\in {0, ..., K}
    \\implies \\alpha_k = 1 / (K + 1) initially.

All other components (graph structure, propagation equation, embedding tables,
BPR objective, evaluation protocol) remain identical to the baseline.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveLightGCN(nn.Module):
    """LightGCN model with learnable softmax layer aggregation weights."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        """Initialize AdaptiveLightGCN model.

        Args:
            num_users: Total number of users M.
            num_items: Total number of items N.
            embedding_dim: Dimensionality of latent embeddings d (default: 64).
            num_layers: Number of graph convolution layers K (default: 3).
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Trainable layer-0 embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Trainable layer aggregation logits theta_0, ..., theta_K
        # Initialized to zeros so softmax(theta)_k = 1 / (K + 1)
        self.layer_logits = nn.Parameter(torch.zeros(num_layers + 1, dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize embeddings using Xavier uniform and layer weights to zero."""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        with torch.no_grad():
            self.layer_logits.zero_()

    def get_layer_weights(self) -> torch.Tensor:
        """Return the current normalized aggregation weights \\alpha_k = softmax(\\theta)_k."""
        return F.softmax(self.layer_logits, dim=0)

    def forward_propagation(
        self, norm_adj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Propagate embeddings across K layers and combine with learned softmax weights.

        Args:
            norm_adj: Symmetrically normalized sparse adjacency tensor \\tilde{A}.

        Returns:
            final_user_emb: Aggregated user embeddings of shape (num_users, embedding_dim).
            final_item_emb: Aggregated item embeddings of shape (num_items, embedding_dim).
            layer_embeddings: List of embedding tensors [E^{(0)}, E^{(1)}, ..., E^{(K)}].
        """
        # Concatenate layer-0 embeddings into E^{(0)}
        e_0 = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

        layer_embeddings: List[torch.Tensor] = [e_0]
        current_emb = e_0

        # Propagate through K layers: E^{(k+1)} = \\tilde{A} E^{(k)}
        for _ in range(self.num_layers):
            current_emb = torch.sparse.mm(norm_adj, current_emb)
            layer_embeddings.append(current_emb)

        # Compute learnable softmax aggregation weights
        # \\alpha_k = softmax(\\theta)_k
        alpha = self.get_layer_weights()  # Shape: (K + 1,)

        # Weighted combination: E^* = \\sum_{k=0}^K \\alpha_k E^{(k)}
        stacked_layers = torch.stack(layer_embeddings, dim=0)  # Shape: (K + 1, M + N, d)
        alpha_weights = alpha.view(self.num_layers + 1, 1, 1)  # Shape: (K + 1, 1, 1)
        final_embeddings = (stacked_layers * alpha_weights).sum(dim=0)  # Shape: (M + N, d)

        final_user_emb = final_embeddings[: self.num_users]
        final_item_emb = final_embeddings[self.num_users :]

        return final_user_emb, final_item_emb, layer_embeddings

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        norm_adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute positive and negative scores along with layer-0 embeddings for BPR loss."""
        final_user_emb, final_item_emb, _ = self.forward_propagation(norm_adj)

        u_final = final_user_emb[users]
        pos_final = final_item_emb[pos_items]
        neg_final = final_item_emb[neg_items]

        # Inner product score: \\hat{y}_{ui} = (e_u^*)^T e_i^*
        pos_scores = (u_final * pos_final).sum(dim=-1)
        neg_scores = (u_final * neg_final).sum(dim=-1)

        u_emb0 = self.user_embedding(users)
        pos_emb0 = self.item_embedding(pos_items)
        neg_emb0 = self.item_embedding(neg_items)

        return pos_scores, neg_scores, u_emb0, pos_emb0, neg_emb0

    def predict(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        norm_adj: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate recommendation scores for explicit (user, item) pairs."""
        final_user_emb, final_item_emb, _ = self.forward_propagation(norm_adj)
        u_emb = final_user_emb[users]
        i_emb = final_item_emb[items]
        return (u_emb * i_emb).sum(dim=-1)

    def predict_all_for_users(
        self,
        users: torch.Tensor,
        norm_adj: torch.Tensor,
    ) -> torch.Tensor:
        """Compute predicted scores for given users across all items."""
        final_user_emb, final_item_emb, _ = self.forward_propagation(norm_adj)
        u_emb = final_user_emb[users]
        scores = torch.matmul(u_emb, final_item_emb.t())
        return scores
