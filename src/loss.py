"""Bayesian Personalized Ranking (BPR) Loss with L2 Regularization.

Paper formulation (Equation 15):
    L_BPR = - sum_{u=1}^M sum_{i in N_u} sum_{j notin N_u} ln sigma(\\hat{y}_{ui} - \\hat{y}_{uj})
            + lambda * ||E^{(0)}||^2

For mini-batch BPR:
    L_bpr = (1 / |B|) * sum_{(u,i,j) in B} -ln sigma(\\hat{y}_{ui} - \\hat{y}_{uj})
    L_reg = (lambda / (2 * |B|)) * sum_{(u,i,j) in B} (||e_u^{(0)}||_2^2 + ||e_i^{(0)}||_2^2 + ||e_j^{(0)}||_2^2)
    L_total = L_bpr + L_reg
"""

from typing import Tuple
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
