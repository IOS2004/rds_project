"""Unit tests for LightGCN model: embedding propagation, layer aggregation, and scoring.

Verifies:
1. One propagation step with manually calculated embeddings.
2. Final embedding aggregation for the original LightGCN (uniform weights 1/(K+1)).
3. Recommendation score calculation via inner product.
4. Matrix formulation equivalence with node-wise formulation.
5. Predict all items candidate ranking shapes and values.
"""

import numpy as np
import torch
import pytest

from src.graph import build_torch_normalized_adjacency
from src.models.lightgcn import LightGCN


def test_manual_propagation_step_and_aggregation():
    """Verify single-step propagation against hand-calculated ground truth.

    Toy Graph:
    Users: u0, u1 (2 users)
    Items: i0, i1 (2 items)
    Edges:
      u0 - i0
      u0 - i1
      u1 - i1

    Degrees:
      d(u0) = 2, d(u1) = 1
      d(i0) = 1, d(i1) = 2

    Normalized adjacency entries:
      (u0, i0): 1 / (sqrt(2) * sqrt(1)) = 1 / sqrt(2)
      (u0, i1): 1 / (sqrt(2) * sqrt(2)) = 1 / 2 = 0.5
      (u1, i1): 1 / (sqrt(1) * sqrt(2)) = 1 / sqrt(2)
    """
    num_users = 2
    num_items = 2
    embedding_dim = 2
    train_pairs = np.array([
        [0, 0],  # u0 - i0
        [0, 1],  # u0 - i1
        [1, 1],  # u1 - i1
    ], dtype=np.int64)

    norm_adj = build_torch_normalized_adjacency(num_users, num_items, train_pairs)

    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        num_layers=1,
    )

    # Set manual layer-0 embeddings
    # User 0: [1.0, 0.0], User 1: [0.0, 1.0]
    # Item 0: [2.0, 0.0], Item 1: [0.0, 2.0]
    with torch.no_grad():
        model.user_embedding.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        )
        model.item_embedding.weight.copy_(
            torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
        )

    # Run forward propagation
    final_user_emb, final_item_emb, layers = model.forward_propagation(norm_adj)

    # Hand-calculated Layer 1 embeddings:
    # e_u0^{(1)} = (1/sqrt(2)) * [2.0, 0.0] + 0.5 * [0.0, 2.0] = [sqrt(2), 1.0]
    # e_u1^{(1)} = (1/sqrt(2)) * [0.0, 2.0] = [0.0, sqrt(2)]
    # e_i0^{(1)} = (1/sqrt(2)) * [1.0, 0.0] = [1/sqrt(2), 0.0]
    # e_i1^{(1)} = 0.5 * [1.0, 0.0] + (1/sqrt(2)) * [0.0, 1.0] = [0.5, 1/sqrt(2)]
    sqrt2 = np.sqrt(2.0)
    expected_e_u0_1 = np.array([sqrt2, 1.0], dtype=np.float32)
    expected_e_u1_1 = np.array([0.0, sqrt2], dtype=np.float32)
    expected_e_i0_1 = np.array([1.0 / sqrt2, 0.0], dtype=np.float32)
    expected_e_i1_1 = np.array([0.5, 1.0 / sqrt2], dtype=np.float32)

    l1 = layers[1].detach().cpu().numpy()
    np.testing.assert_allclose(l1[0], expected_e_u0_1, rtol=1e-5)
    np.testing.assert_allclose(l1[1], expected_e_u1_1, rtol=1e-5)
    np.testing.assert_allclose(l1[2], expected_e_i0_1, rtol=1e-5)
    np.testing.assert_allclose(l1[3], expected_e_i1_1, rtol=1e-5)

    # Hand-calculated Aggregated Embeddings (K=1 => uniform weights alpha = 1/2):
    # e_u0^* = 0.5 * ([1.0, 0.0] + [sqrt(2), 1.0]) = [(1 + sqrt(2))/2, 0.5]
    # e_i0^* = 0.5 * ([2.0, 0.0] + [1/sqrt(2), 0.0]) = [(2 + 1/sqrt(2))/2, 0.0]
    expected_u0_final = 0.5 * (np.array([1.0, 0.0], dtype=np.float32) + expected_e_u0_1)
    expected_i0_final = 0.5 * (np.array([2.0, 0.0], dtype=np.float32) + expected_e_i0_1)

    np.testing.assert_allclose(final_user_emb[0].detach().cpu().numpy(), expected_u0_final, rtol=1e-5)
    np.testing.assert_allclose(final_item_emb[0].detach().cpu().numpy(), expected_i0_final, rtol=1e-5)


def test_recommendation_scoring():
    """Verify dot product scoring \\hat{y}_{ui} = (e_u^*)^T e_i^*."""
    num_users = 2
    num_items = 3
    embedding_dim = 4
    train_pairs = np.array([[0, 0], [1, 1], [0, 2]], dtype=np.int64)

    norm_adj = build_torch_normalized_adjacency(num_users, num_items, train_pairs)
    model = LightGCN(num_users, num_items, embedding_dim=embedding_dim, num_layers=2)

    users = torch.tensor([0, 1], dtype=torch.long)
    pos_items = torch.tensor([0, 1], dtype=torch.long)
    neg_items = torch.tensor([1, 2], dtype=torch.long)

    pos_scores, neg_scores, u0, pos0, neg0 = model(users, pos_items, neg_items, norm_adj)

    assert pos_scores.shape == (2,)
    assert neg_scores.shape == (2,)
    assert u0.shape == (2, embedding_dim)
    assert pos0.shape == (2, embedding_dim)
    assert neg0.shape == (2, embedding_dim)

    # Predict all candidates
    all_scores = model.predict_all_for_users(users, norm_adj)
    assert all_scores.shape == (2, num_items)
    # Check that score for (u0, item 0) matches pos_scores[0]
    np.testing.assert_allclose(
        all_scores[0, 0].item(), pos_scores[0].item(), rtol=1e-5
    )
