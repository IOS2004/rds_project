"""Bayesian Personalized Ranking (BPR) Loss with L2 Regularization.

Paper formulation (Equation 15):
    L_BPR = - sum_{u=1}^M sum_{i in N_u} sum_{j notin N_u} ln sigma(\\hat{y}_{ui} - \\hat{y}_{uj})
            + lambda * ||E^{(0)}||^2

For mini-batch BPR:
    L_bpr = (1 / |B|) * sum_{(u,i,j) in B} -ln sigma(\\hat{y}_{ui} - \\hat{y}_{uj})
    L_reg = (lambda / (2 * |B|)) * sum_{(u,i,j) in B} (||e_u^{(0)}||_2^2 + ||e_i^{(0)}||_2^2 + ||e_j^{(0)}||_2^2)
    L_total = L_bpr + L_reg
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    """Pairwise BPR ranking loss with embedding L2 regularization."""

    def __init__(self, reg_weight: float = 1e-4) -> None:
        """Initialize BPR loss.

        Args:
            reg_weight: L2 regularization strength lambda (default: 1e-4, optimal in paper).
        """
        super().__init__()
        self.reg_weight = float(reg_weight)

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        u_emb0: torch.Tensor,
        pos_emb0: torch.Tensor,
        neg_emb0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute BPR loss and embedding L2 regularization.

        Args:
            pos_scores: 1D tensor of predicted scores for observed pairs (\\hat{y}_{ui}).
            neg_scores: 1D tensor of predicted scores for unobserved pairs (\\hat{y}_{uj}).
            u_emb0: Layer-0 embeddings of target users in the batch.
            pos_emb0: Layer-0 embeddings of positive items in the batch.
            neg_emb0: Layer-0 embeddings of sampled negative items in the batch.

        Returns:
            total_loss: L_total = L_bpr + L_reg.
            bpr_loss: Mean negative log-sigmoid of score margin.
            reg_loss: Scaled L2 regularization on layer-0 embeddings.
        """
        # Score difference: x_uij = \\hat{y}_{ui} - \\hat{y}_{uj}
        score_diff = pos_scores - neg_scores

        # Numerically stable -ln sigma(x) = softplus(-x)
        bpr_loss = -F.logsigmoid(score_diff).mean()

        # L2 regularization on layer-0 embeddings: ||e_u^{(0)}||^2 + ||e_i^{(0)}||^2 + ||e_j^{(0)}||^2
        batch_size = pos_scores.shape[0]
        if batch_size > 0 and self.reg_weight > 0.0:
            reg_term = (
                torch.sum(u_emb0.pow(2))
                + torch.sum(pos_emb0.pow(2))
                + torch.sum(neg_emb0.pow(2))
            )
            reg_loss = self.reg_weight * (reg_term / (2.0 * float(batch_size)))
        else:
            reg_loss = torch.tensor(0.0, device=pos_scores.device, dtype=pos_scores.dtype)

        total_loss = bpr_loss + reg_loss
        return total_loss, bpr_loss, reg_loss


class CosineContrastiveLoss(nn.Module):
    """Cosine Contrastive Loss (CCL) with L2 Regularization.

    Based on SimpleX paper:
        "SimpleX: A Simple and Strong Baseline for Collaborative Filtering"
        Kelong Mao et al., CIKM 2021 (arXiv:2109.12613).

    Paper formulation:
        L_CCL(u, i) = (1 - y_ui) + (w / |N|) * sum_{j in N} max(0, y_uj - m)

    where:
        - y_ui is the cosine similarity between user u and positive item i.
        - y_uj is the cosine similarity between user u and negative item j.
        - m is the negative-similarity margin threshold.
        - w is the negative weight hyperparameter.
        - N is the set of sampled negative items for that positive interaction.

    Regularization:
        L_reg = (lambda / (2 * |B|)) * sum_{b in B} (
            ||e_u^{(0)}||_2^2 + ||e_i^{(0)}||_2^2 + (1 / |N|) * sum_{j in N} ||e_j^{(0)}||_2^2
        )
        L_total = L_ccl + L_reg
    """

    def __init__(
        self,
        margin: float = 0.8,
        negative_weight: float = 10.0,
        reg_weight: float = 1e-4,
        eps: float = 1e-8,
    ) -> None:
        """Initialize Cosine Contrastive Loss.

        Args:
            margin: Margin threshold m for negative item cosine similarities (default: 0.8).
                    Negative items with similarity <= m incur zero penalty.
            negative_weight: Weight w controlling contribution of negative penalty (default: 10.0).
            reg_weight: L2 regularization strength lambda on initial layer-0 embeddings (default: 1e-4).
            eps: Numerical stability constant for cosine normalization (default: 1e-8).
        """
        super().__init__()
        self.margin = float(margin)
        self.negative_weight = float(negative_weight)
        self.reg_weight = float(reg_weight)
        self.eps = float(eps)

    def loss_from_similarities(
        self,
        pos_sim: torch.Tensor,
        neg_sim: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute CCL loss directly from precomputed cosine similarities.

        Args:
            pos_sim: 1D tensor of shape (batch_size,) with positive cosine similarities y_ui.
            neg_sim: Tensor of shape (batch_size, num_negatives) or (batch_size,)
                     with negative cosine similarities y_uj.

        Returns:
            ccl_loss: Mean CCL loss across the batch.
            avg_neg_penalty: Mean raw negative penalty before w weighting.
        """
        # Ensure pos_sim is 1D
        if pos_sim.dim() > 1:
            pos_sim = pos_sim.squeeze(-1)

        # Positive loss: 1 - y_ui
        pos_loss = 1.0 - pos_sim

        # Ensure neg_sim is 2D: (batch_size, num_negatives)
        if neg_sim.dim() == 1:
            neg_sim = neg_sim.unsqueeze(1)

        # Negative penalty: max(0, y_uj - m) = ReLU(y_uj - m)
        neg_penalty = F.relu(neg_sim - self.margin)

        # Average penalty over sampled negatives: (1 / |N|) * sum_{j in N} max(0, y_uj - m)
        avg_neg_penalty = neg_penalty.mean(dim=-1)

        # Example-level CCL loss: (1 - y_ui) + w * avg_neg_penalty
        example_ccl = pos_loss + self.negative_weight * avg_neg_penalty

        # Batch mean
        ccl_loss = example_ccl.mean()
        return ccl_loss, avg_neg_penalty.mean()

    def forward(
        self,
        u_final: torch.Tensor,
        pos_final: torch.Tensor,
        neg_final: torch.Tensor,
        u_emb0: Optional[torch.Tensor] = None,
        pos_emb0: Optional[torch.Tensor] = None,
        neg_emb0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute CCL loss from final embeddings and L2 regularization on layer-0 embeddings.

        Args:
            u_final: Final user embeddings of shape (batch_size, embedding_dim).
            pos_final: Final positive item embeddings of shape (batch_size, embedding_dim).
            neg_final: Final negative item embeddings of shape (batch_size, num_negatives, embedding_dim)
                       or (batch_size, embedding_dim).
            u_emb0: Layer-0 user embeddings of shape (batch_size, embedding_dim).
            pos_emb0: Layer-0 positive item embeddings of shape (batch_size, embedding_dim).
            neg_emb0: Layer-0 negative item embeddings of shape (batch_size, num_negatives, embedding_dim)
                      or (batch_size, embedding_dim).

        Returns:
            total_loss: L_total = L_ccl + L_reg.
            ccl_loss: Cosine contrastive loss term.
            reg_loss: Embedding L2 regularization term.
        """
        batch_size = u_final.shape[0]
        if batch_size == 0:
            zero = torch.tensor(0.0, device=u_final.device, dtype=u_final.dtype)
            return zero, zero, zero

        # Numerically stable L2 normalization with eps clamp to protect against zero-norm
        u_norm = F.normalize(u_final, p=2, dim=-1, eps=self.eps)  # (B, d)
        pos_norm = F.normalize(pos_final, p=2, dim=-1, eps=self.eps)  # (B, d)

        # Positive cosine similarity y_ui = cos(u, pos)
        pos_sim = (u_norm * pos_norm).sum(dim=-1)  # (B,)

        # Negative cosine similarity y_uj = cos(u, neg)
        if neg_final.dim() == 2:
            neg_final = neg_final.unsqueeze(1)  # (B, 1, d)

        neg_norm = F.normalize(neg_final, p=2, dim=-1, eps=self.eps)  # (B, N, d)
        # u_norm: (B, 1, d) * neg_norm: (B, N, d) -> sum(-1) -> (B, N)
        neg_sim = (u_norm.unsqueeze(1) * neg_norm).sum(dim=-1)

        ccl_loss, _ = self.loss_from_similarities(pos_sim, neg_sim)

        # L2 Regularization on initial layer-0 embeddings
        # Consistent with BPR regularizer, normalized by number of negatives to preserve energy scale
        if (
            self.reg_weight > 0.0
            and u_emb0 is not None
            and pos_emb0 is not None
            and neg_emb0 is not None
        ):
            num_neg = neg_emb0.shape[1] if neg_emb0.dim() == 3 else 1
            reg_term = (
                torch.sum(u_emb0.pow(2))
                + torch.sum(pos_emb0.pow(2))
                + torch.sum(neg_emb0.pow(2)) / float(num_neg)
            )
            reg_loss = self.reg_weight * (reg_term / (2.0 * float(batch_size)))
        else:
            reg_loss = torch.tensor(0.0, device=u_final.device, dtype=u_final.dtype)

        total_loss = ccl_loss + reg_loss
        return total_loss, ccl_loss, reg_loss

