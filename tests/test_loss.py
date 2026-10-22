"""Unit tests for BPR loss behavior and L2 regularization."""

import torch
import pytest

from src.loss import BPRLoss


def test_bpr_loss_ordering():
    """Verify BPR loss behavior for correctly vs incorrectly ranked pairs."""
    loss_fn = BPRLoss(reg_weight=0.0)

    # Correct ranking: pos_score > neg_score
    pos_scores_correct = torch.tensor([5.0, 3.0, 4.0])
    neg_scores_correct = torch.tensor([1.0, 0.0, -1.0])

    dummy_emb = torch.zeros((3, 4))
    total_corr, bpr_corr, _ = loss_fn(
        pos_scores_correct, neg_scores_correct, dummy_emb, dummy_emb, dummy_emb
    )

    # Inverted ranking: pos_score < neg_score
    pos_scores_inverted = torch.tensor([1.0, 0.0, -1.0])
    neg_scores_inverted = torch.tensor([5.0, 3.0, 4.0])

    total_inv, bpr_inv, _ = loss_fn(
        pos_scores_inverted, neg_scores_inverted, dummy_emb, dummy_emb, dummy_emb
    )

    # Correct ranking must have significantly lower loss than inverted ranking
    assert bpr_corr.item() < bpr_inv.item()

    # When pos is much larger than neg, -log(sigmoid(margin)) approaches 0
    pos_large = torch.tensor([50.0])
    neg_small = torch.tensor([-50.0])
    _, bpr_zero, _ = loss_fn(pos_large, neg_small, dummy_emb[:1], dummy_emb[:1], dummy_emb[:1])
    assert bpr_zero.item() < 1e-6


def test_embedding_regularization():
    """Verify L2 embedding regularization calculation."""
    reg_weight = 1e-2
    loss_fn = BPRLoss(reg_weight=reg_weight)

    pos_scores = torch.tensor([1.0, 1.0])
    neg_scores = torch.tensor([0.0, 0.0])

    # Batch size = 2, dim = 2
    u0 = torch.tensor([[1.0, 1.0], [2.0, 2.0]])      # sum sq = 1+1+4+4 = 10
    pos0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])    # sum sq = 1+1 = 2
    neg0 = torch.tensor([[0.0, 2.0], [2.0, 0.0]])    # sum sq = 4+4 = 8
    # Total sum of squares = 10 + 2 + 8 = 20
    # Expected reg_loss = reg_weight * (20 / (2 * 2)) = 0.01 * 5.0 = 0.05

    _, _, reg_loss = loss_fn(pos_scores, neg_scores, u0, pos0, neg0)
    assert pytest.approx(reg_loss.item(), rel=1e-5) == 0.05
