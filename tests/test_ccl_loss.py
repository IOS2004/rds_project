"""Unit tests for Cosine Contrastive Loss (CCL) mathematics and properties.

Verifies:
1. Exact hand-calculated numerical example with known similarities.
2. Below-margin negative sample contributes zero penalty.
3. Above-margin negative sample contributes the exact expected penalty.
4. Negative penalty is averaged over the exact number of sampled negative items |N|.
5. Numerical stability and zero-norm vector protection (no NaNs or Infs).
6. Gradients flow through CCL back to trainable initial LightGCN embeddings.
7. Layer-0 embedding regularization calculation.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss import CosineContrastiveLoss
from src.models.lightgcn_ccl import LightGCNCCL
from src.graph import build_torch_normalized_adjacency


def test_ccl_hand_calculated_example():
    """Verify CCL against an exact analytical hand calculation.

    Setup:
        pos_sim = 0.8
        neg_sim = [0.2, 0.6]  (|N| = 2)
        margin m = 0.4
        negative_weight w = 10.0

    Calculation:
        L_pos = 1 - 0.8 = 0.2
        neg_1 = 0.2 <= 0.4 -> penalty = 0.0
        neg_2 = 0.6 > 0.4  -> penalty = 0.6 - 0.4 = 0.2
        avg_neg_penalty = (0.0 + 0.2) / 2 = 0.1
        L_ccl = L_pos + w * avg_neg_penalty = 0.2 + 10.0 * 0.1 = 1.2
    """
    ccl = CosineContrastiveLoss(margin=0.4, negative_weight=10.0, reg_weight=0.0)

    pos_sim = torch.tensor([0.8], dtype=torch.float32)
    neg_sim = torch.tensor([[0.2, 0.6]], dtype=torch.float32)

    ccl_loss, avg_penalty = ccl.loss_from_similarities(pos_sim, neg_sim)

    expected_loss = 1.2
    expected_penalty = 0.1

    assert torch.isclose(ccl_loss, torch.tensor(expected_loss, dtype=torch.float32), atol=1e-6)
    assert torch.isclose(avg_penalty, torch.tensor(expected_penalty, dtype=torch.float32), atol=1e-6)


def test_ccl_below_margin_zero_penalty():
    """Verify that negative pairs with cosine similarity below the margin incur strictly zero penalty."""
    margin = 0.5
    ccl = CosineContrastiveLoss(margin=margin, negative_weight=10.0, reg_weight=0.0)

    # All negative similarities strictly below margin
    pos_sim = torch.tensor([1.0], dtype=torch.float32)  # pos_loss = 0
    neg_sim = torch.tensor([[-0.5, 0.0, 0.3, 0.49]], dtype=torch.float32)

    ccl_loss, avg_penalty = ccl.loss_from_similarities(pos_sim, neg_sim)

    assert torch.isclose(avg_penalty, torch.tensor(0.0), atol=1e-7)
    assert torch.isclose(ccl_loss, torch.tensor(0.0), atol=1e-7)


def test_ccl_above_margin_expected_penalty():
    """Verify that negative pairs with similarity above the margin incur the exact linear penalty."""
    margin = 0.3
    ccl = CosineContrastiveLoss(margin=margin, negative_weight=5.0, reg_weight=0.0)

    pos_sim = torch.tensor([1.0], dtype=torch.float32)  # pos_loss = 0
    neg_sim = torch.tensor([[0.7]], dtype=torch.float32)  # |N| = 1

    ccl_loss, avg_penalty = ccl.loss_from_similarities(pos_sim, neg_sim)

    expected_penalty = 0.7 - 0.3  # 0.4
    expected_loss = 5.0 * expected_penalty  # 2.0

    assert torch.isclose(avg_penalty, torch.tensor(expected_penalty), atol=1e-6)
    assert torch.isclose(ccl_loss, torch.tensor(expected_loss), atol=1e-6)


def test_ccl_negative_averaging_over_correct_count():
    """Verify that negative penalties are strictly averaged by |N|, not summed."""
    margin = 0.2
    w = 1.0
    ccl = CosineContrastiveLoss(margin=margin, negative_weight=w, reg_weight=0.0)

    pos_sim = torch.tensor([1.0], dtype=torch.float32)

    # 4 negatives: two above margin, two below
    # [0.4 - 0.2 = 0.2, 0.6 - 0.2 = 0.4, 0.0 <= 0.2 -> 0, -0.1 <= 0.2 -> 0]
    # sum = 0.6. Average = 0.6 / 4 = 0.15
    neg_sim = torch.tensor([[0.4, 0.6, 0.0, -0.1]], dtype=torch.float32)

    ccl_loss, avg_penalty = ccl.loss_from_similarities(pos_sim, neg_sim)

    assert torch.isclose(avg_penalty, torch.tensor(0.15), atol=1e-6)
    assert torch.isclose(ccl_loss, torch.tensor(0.15), atol=1e-6)


def test_ccl_numerical_stability_zero_norm():
    """Verify numerical stability with zero or near-zero norm embedding vectors."""
    ccl = CosineContrastiveLoss(margin=0.5, negative_weight=10.0, eps=1e-8)

    # Zero vector representation
    u_final = torch.zeros(2, 8, dtype=torch.float32)
    pos_final = torch.zeros(2, 8, dtype=torch.float32)
    neg_final = torch.zeros(2, 5, 8, dtype=torch.float32)

    total_loss, ccl_loss, reg_loss = ccl(u_final, pos_final, neg_final)

    assert not torch.isnan(total_loss)
    assert not torch.isnan(ccl_loss)
    assert not torch.isinf(total_loss)
    assert not torch.isinf(ccl_loss)


def test_ccl_gradient_flow_to_initial_embeddings():
    """Verify end-to-end backpropagation through LightGCN propagation to initial embeddings."""
    num_users = 4
    num_items = 6
    d = 16
    K = 2

    # Simple toy bipartite graph
    train_pairs = [[0, 0], [0, 1], [1, 2], [2, 3], [3, 4], [3, 5]]
    norm_adj = build_torch_normalized_adjacency(num_users, num_items, train_pairs)

    model = LightGCNCCL(num_users, num_items, embedding_dim=d, num_layers=K)
    ccl = CosineContrastiveLoss(margin=0.5, negative_weight=5.0, reg_weight=1e-4)

    users = torch.tensor([0, 1], dtype=torch.long)
    pos_items = torch.tensor([1, 2], dtype=torch.long)
    neg_items = torch.tensor([[3, 4, 5], [0, 1, 3]], dtype=torch.long)  # (2, 3)

    u_final, pos_final, neg_final, u0, pos0, neg0 = model(users, pos_items, neg_items, norm_adj)
    total_loss, ccl_loss, reg_loss = ccl(u_final, pos_final, neg_final, u0, pos0, neg0)

    total_loss.backward()

    # Verify gradients exist and are non-zero for initial layer-0 embeddings
    assert model.user_embedding.weight.grad is not None
    assert model.item_embedding.weight.grad is not None
    assert model.user_embedding.weight.grad.abs().sum().item() > 0.0
    assert model.item_embedding.weight.grad.abs().sum().item() > 0.0
    assert not torch.isnan(model.user_embedding.weight.grad).any()
    assert not torch.isnan(model.item_embedding.weight.grad).any()


def test_ccl_regularization_scaling():
    """Verify that L2 regularization scales correctly with negative sample count."""
    reg_weight = 1e-4
    ccl = CosineContrastiveLoss(margin=0.5, negative_weight=1.0, reg_weight=reg_weight)

    B = 2
    d = 4
    N_neg = 3

    u_final = torch.ones(B, d)
    pos_final = torch.ones(B, d)
    neg_final = torch.ones(B, N_neg, d)

    u0 = torch.full((B, d), 2.0)
    pos0 = torch.full((B, d), 3.0)
    neg0 = torch.full((B, N_neg, d), 4.0)

    # ||u0||^2 = B * d * 4 = 2 * 4 * 4 = 32
    # ||pos0||^2 = B * d * 9 = 2 * 4 * 9 = 72
    # ||neg0||^2 = B * N_neg * d * 16 = 2 * 3 * 4 * 16 = 384
    # With normalization by N_neg: ||neg0||^2 / N_neg = 384 / 3 = 128
    # reg_term = 32 + 72 + 128 = 232
    # reg_loss = reg_weight * (232 / (2 * B)) = 1e-4 * (232 / 4) = 5.8e-3

    _, _, reg_loss = ccl(u_final, pos_final, neg_final, u0, pos0, neg0)
    expected_reg = reg_weight * (232.0 / (2.0 * B))

    assert torch.isclose(reg_loss, torch.tensor(expected_reg), atol=1e-7)
