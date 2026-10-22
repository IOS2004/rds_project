"""Original LightGCN Model Implementation.

Faithfully implements:
“LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation”
Xiangnan He et al., SIGIR 2020 (arXiv:2002.02126).

Key equations:
1. Propagation:
       E^{(k+1)} = \\tilde{A} E^{(k)}
   where \\tilde{A} = D^{-1/2} A D^{-1/2} is the symmetrically normalized adjacency matrix.
2. Layer combination (uniform fixed aggregation):
       E^* = \\frac{1}{K + 1} \\sum_{k=0}^{K} E^{(k)}
3. Dot-product recommendation scoring:
       \\hat{y}_{ui} = (e_u^*)^T e_i^*
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn


class LightGCN(nn.Module):
    """Original LightGCN model with fixed uniform layer aggregation."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        """Initialize LightGCN model.

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

        # Trainable layer-0 embeddings E^{(0)}: user embedding P, item embedding Q
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize embeddings using Xavier uniform initialization as specified in paper."""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward_propagation(
        self, norm_adj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Propagate embeddings across K layers and combine with uniform weights.

        Args:
            norm_adj: Symmetrically normalized sparse adjacency tensor \\tilde{A}
                      of shape (M + N, M + N).

        Returns:
            final_user_emb: Aggregated user embeddings of shape (num_users, embedding_dim).
            final_item_emb: Aggregated item embeddings of shape (num_items, embedding_dim).
            layer_embeddings: List of embedding tensors [E^{(0)}, E^{(1)}, ..., E^{(K)}],
                              where each tensor has shape (M + N, embedding_dim).
        """
        # Concatenate layer-0 user and item embeddings into E^{(0)}
        e_0 = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

        layer_embeddings: List[torch.Tensor] = [e_0]
        current_emb = e_0

        # Propagate through K layers: E^{(k+1)} = \\tilde{A} E^{(k)}
        for _ in range(self.num_layers):
            current_emb = torch.sparse.mm(norm_adj, current_emb)
            layer_embeddings.append(current_emb)

        # Baseline uniform aggregation: E^* = 1 / (K + 1) * \\sum_{k=0}^K E^{(k)}
        stacked_layers = torch.stack(layer_embeddings, dim=0)  # Shape: (K+1, M+N, d)
        final_embeddings = torch.mean(stacked_layers, dim=0)    # Shape: (M+N, d)

        # Separate user and item embeddings
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
        """Compute positive and negative scores along with layer-0 embeddings for BPR loss.

        Args:
            users: 1D tensor of user IDs.
            pos_items: 1D tensor of positive item IDs.
            neg_items: 1D tensor of sampled negative item IDs.
            norm_adj: Symmetrically normalized sparse adjacency tensor.

        Returns:
            pos_scores: Dot-product scores for observed user-item pairs (\\hat{y}_{ui}).
            neg_scores: Dot-product scores for unobserved user-item pairs (\\hat{y}_{uj}).
            u_emb0: Layer-0 user embeddings for regularizing the batch.
            pos_emb0: Layer-0 positive item embeddings for regularizing the batch.
            neg_emb0: Layer-0 negative item embeddings for regularizing the batch.
        """
        final_user_emb, final_item_emb, _ = self.forward_propagation(norm_adj)

        # Look up aggregated representations
        u_final = final_user_emb[users]
        pos_final = final_item_emb[pos_items]
        neg_final = final_item_emb[neg_items]

        # Inner product score: \\hat{y}_{ui} = (e_u^*)^T e_i^*
        pos_scores = (u_final * pos_final).sum(dim=-1)
        neg_scores = (u_final * neg_final).sum(dim=-1)

        # Retrieve layer-0 embeddings for BPR L2 regularization
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
        """Compute predicted scores for given users across all N items.

        Args:
            users: 1D tensor of user IDs.
            norm_adj: Symmetrically normalized sparse adjacency tensor.

        Returns:
            scores: Tensor of shape (len(users), num_items), where
                    scores[b, i] is the score of user[b] on item i.
        """
        final_user_emb, final_item_emb, _ = self.forward_propagation(norm_adj)
        u_emb = final_user_emb[users]  # Shape: (B, d)
        # Dot product with all items: (B, d) @ (d, N) -> (B, N)
        scores = torch.matmul(u_emb, final_item_emb.t())
        return scores
