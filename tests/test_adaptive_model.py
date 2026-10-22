"""Unit tests for AdaptiveLightGCN: uniform initialization, gradient flow, and baseline equivalence."""

import numpy as np
import torch
import pytest

from src.graph import build_torch_normalized_adjacency
from src.models.lightgcn import LightGCN
from src.models.adaptive_lightgcn import AdaptiveLightGCN
from src.loss import BPRLoss


def test_uniform_initialization_of_adaptive_weights():
    """Verify that AdaptiveLightGCN initializes with exactly uniform weights: 1 / (K + 1)."""
    num_users = 5
    num_items = 6
    num_layers = 3  # K = 3 => K + 1 = 4 layers (0, 1, 2, 3)

    model = AdaptiveLightGCN(num_users, num_items, embedding_dim=16, num_layers=num_layers)

    # Initial layer_logits must be all zeros
    np.testing.assert_array_equal(model.layer_logits.detach().cpu().numpy(), np.zeros(4))

    # Initial weights after softmax must equal 1 / 4 = 0.25
    weights = model.get_layer_weights().detach().cpu().numpy()
    expected_weights = np.full(4, 0.25, dtype=np.float32)
    np.testing.assert_allclose(weights, expected_weights, rtol=1e-6)


def test_gradient_flow_to_adaptive_weights():
    """Verify that backpropagation computes valid non-zero gradients for layer_logits."""
    num_users = 4
    num_items = 5
    num_layers = 2
    train_pairs = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int64)

    norm_adj = build_torch_normalized_adjacency(num_users, num_items, train_pairs)
    model = AdaptiveLightGCN(num_users, num_items, embedding_dim=8, num_layers=num_layers)
    loss_fn = BPRLoss(reg_weight=1e-4)

    users = torch.tensor([0, 1], dtype=torch.long)
    pos_items = torch.tensor([0, 1], dtype=torch.long)
    neg_items = torch.tensor([2, 3], dtype=torch.long)

    pos_scores, neg_scores, u0, pos0, neg0 = model(users, pos_items, neg_items, norm_adj)
    loss, _, _ = loss_fn(pos_scores, neg_scores, u0, pos0, neg0)

    # Before backward, grad is None
    assert model.layer_logits.grad is None

    loss.backward()

    # After backward, grad must exist, be finite, and have non-zero elements
    assert model.layer_logits.grad is not None
    grad = model.layer_logits.grad.detach().cpu().numpy()
    assert not np.isnan(grad).any()
    assert not np.isinf(grad).any()
    assert np.any(np.abs(grad) > 1e-7), f"Gradients were unexpectedly zero: {grad}"


def test_baseline_equivalence_at_initialization():
    """Verify that at initialization, AdaptiveLightGCN produces identical predictions to LightGCN."""
    num_users = 3
    num_items = 3
    num_layers = 2
    train_pairs = np.array([[0, 0], [1, 1], [0, 2]], dtype=np.int64)

    norm_adj = build_torch_normalized_adjacency(num_users, num_items, train_pairs)

    baseline = LightGCN(num_users, num_items, embedding_dim=4, num_layers=num_layers)
    adaptive = AdaptiveLightGCN(num_users, num_items, embedding_dim=4, num_layers=num_layers)

    # Copy identical embedding weights
    with torch.no_grad():
        adaptive.user_embedding.weight.copy_(baseline.user_embedding.weight)
        adaptive.item_embedding.weight.copy_(baseline.item_embedding.weight)

    u_base, i_base, _ = baseline.forward_propagation(norm_adj)
    u_adapt, i_adapt, _ = adaptive.forward_propagation(norm_adj)

    np.testing.assert_allclose(
        u_base.detach().cpu().numpy(),
        u_adapt.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        i_base.detach().cpu().numpy(),
        i_adapt.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
