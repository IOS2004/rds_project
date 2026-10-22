"""LightGCNCCL: LightGCN with Cosine Contrastive Loss (CCL).

Based on the approach described in:
"SimpleX: A Simple and Strong Baseline for Collaborative Filtering"
Kelong Mao et al., CIKM 2021 (arXiv:2109.12613).

This variant retains the exact LightGCN graph propagation and uniform layer aggregation:
    E^{(k+1)} = \\tilde{A} E^{(k)}
    E^* = \\frac{1}{K + 1} \\sum_{k=0}^K E^{(k)}

and pairs it with the Cosine Contrastive Loss (CCL) objective:
    L_CCL(u, i) = (1 - y_ui) + (w / |N|) * \\sum_{j in N} max(0, y_uj - m)

where y_ui and y_uj are cosine similarities computed on the aggregated representations E^*,
m is the negative margin, w is the negative weight, and N is the set of sampled negative items.
Recommendation scoring defaults to cosine similarity for consistency with CCL.
"""

from typing import Tuple, Optional
import torch
import torch.nn.functional as F

from src.models.lightgcn import LightGCN


class LightGCNCCL(LightGCN):
    """LightGCN model adapted for Cosine Contrastive Loss (SimpleX paper)."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        score_type: str = "cosine",
    ) -> None:
        """Initialize LightGCNCCL model.

        Args:
            num_users: Total number of users M.
            num_items: Total number of items N.
            embedding_dim: Dimensionality of latent embeddings d (default: 64).
            num_layers: Number of graph convolution layers K (default: 3).
            score_type: Scoring function for ranking and evaluation ("cosine" or "dot", default: "cosine").
        """
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
        )
        if score_type not in ("cosine", "dot"):
            raise ValueError(f"score_type must be 'cosine' or 'dot', got {score_type}")
        self.score_type = score_type

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        norm_adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute final and layer-0 embeddings for Cosine Contrastive Loss.

        Args:
            users: 1D tensor of user IDs of shape (batch_size,).
            pos_items: 1D tensor of positive item IDs of shape (batch_size,).
            neg_items: Tensor of sampled negative item IDs:
                       - shape (batch_size, num_negatives) for multi-negative sampling
                       - shape (batch_size,) for single-negative sampling
            norm_adj: Symmetrically normalized sparse adjacency tensor.

        Returns:
            u_final: Final user representations of shape (batch_size, embedding_dim).
            pos_final: Final positive item representations of shape (batch_size, embedding_dim).
            neg_final: Final negative item representations of shape
                       (batch_size, num_negatives, embedding_dim) or (batch_size, embedding_dim).
            u_emb0: Layer-0 user embeddings of shape (batch_size, embedding_dim).
            pos_emb0: Layer-0 positive item embeddings of shape (batch_size, embedding_dim).
            neg_emb0: Layer-0 negative item embeddings of shape
                      (batch_size, num_negatives, embedding_dim) or (batch_size, embedding_dim).
        """
        final_user_emb, final_item_emb, _ = self.forward_propagation(norm_adj)

        u_final = final_user_emb[users]
        pos_final = final_item_emb[pos_items]
        neg_final = final_item_emb[neg_items]

        u_emb0 = self.user_embedding(users)
        pos_emb0 = self.item_embedding(pos_items)
        neg_emb0 = self.item_embedding(neg_items)

        return u_final, pos_final, neg_final, u_emb0, pos_emb0, neg_emb0

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

        if self.score_type == "cosine":
            return F.cosine_similarity(u_emb, i_emb, dim=-1, eps=1e-8)
        return (u_emb * i_emb).sum(dim=-1)

    def predict_all_for_users(
        self,
        users: torch.Tensor,
        norm_adj: torch.Tensor,
    ) -> torch.Tensor:
        """Compute predicted scores for given users across all N items.

        If score_type == "cosine", uses L2-normalized embeddings so dot products
        yield exact cosine similarities.
        """
        final_user_emb, final_item_emb, _ = self.forward_propagation(norm_adj)
        u_emb = final_user_emb[users]

        if self.score_type == "cosine":
            u_norm = F.normalize(u_emb, p=2, dim=-1, eps=1e-8)
            i_norm = F.normalize(final_item_emb, p=2, dim=-1, eps=1e-8)
            return torch.matmul(u_norm, i_norm.t())

        return torch.matmul(u_emb, final_item_emb.t())
