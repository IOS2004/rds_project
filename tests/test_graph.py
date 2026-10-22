"""Unit tests for user-item bipartite graph construction and degree normalization.

Verifies:
1. User-item graph construction from a hand-written interaction dataset.
2. Adjacency symmetry: A = A^T.
3. Expected node degrees matching theoretical counts.
4. Degree-based normalization on a toy graph with hand-calculated ground truth.
5. Safe handling of isolated nodes (degree 0) with zero division protection.
6. Equivalence between SciPy sparse matrix and PyTorch sparse tensor representations.
"""

import numpy as np
import scipy.sparse as sp
import torch
import pytest

from src.graph import (
    compute_node_degrees,
    build_sparse_adjacency,
    build_normalized_adjacency,
    scipy_coo_to_torch_sparse,
    build_torch_normalized_adjacency,
)


def test_hand_crafted_graph_construction_and_symmetry():
    """Test 1 & 2: User-item graph construction, symmetry, and node degrees."""
    # 2 users (u0, u1) and 3 items (i0, i1, i2)
    # i2 is deliberately left isolated (no interactions)
    num_users = 2
    num_items = 3
    train_pairs = np.array([
        [0, 0],  # u0 - i0
        [0, 1],  # u0 - i1
        [1, 1],  # u1 - i1
    ], dtype=np.int64)

    adj = build_sparse_adjacency(num_users, num_items, train_pairs)
    total_nodes = num_users + num_items
    assert adj.shape == (total_nodes, total_nodes)

    # Convert to dense for structural validation
    dense_adj = adj.toarray()

    # 1. Symmetry check: A must equal A^T
    np.testing.assert_array_equal(dense_adj, dense_adj.T)

    # 2. Block bipartite check: Diagonal blocks (user-user and item-item) must be zero
    user_block = dense_adj[:num_users, :num_users]
    item_block = dense_adj[num_users:, num_users:]
    np.testing.assert_array_equal(user_block, 0)
    np.testing.assert_array_equal(item_block, 0)

    # 3. Off-diagonal block R and R^T check
    r_block = dense_adj[:num_users, num_users:]
    expected_r = np.array([
        [1, 1, 0],  # u0 interacted with i0, i1
        [0, 1, 0],  # u1 interacted with i1
    ], dtype=np.float32)
    np.testing.assert_array_equal(r_block, expected_r)

    # 4. Degree computation check
    u_degs, i_degs, total_degs = compute_node_degrees(num_users, num_items, train_pairs)
    np.testing.assert_array_equal(u_degs, np.array([2.0, 1.0]))
    np.testing.assert_array_equal(i_degs, np.array([1.0, 2.0, 0.0]))
    np.testing.assert_array_equal(total_degs, np.array([2.0, 1.0, 1.0, 2.0, 0.0]))


def test_degree_normalization_toy_graph():
    """Test 3 & 4: Degree normalization \\tilde{A} = D^{-1/2} A D^{-1/2} on toy graph."""
    num_users = 2
    num_items = 3
    train_pairs = np.array([
        [0, 0],  # u0 - i0
        [0, 1],  # u0 - i1
        [1, 1],  # u1 - i1
    ], dtype=np.int64)

    # Theoretical calculations:
    # Node indices:
    # u0 -> 0, u1 -> 1
    # i0 -> 2, i1 -> 3, i2 -> 4
    # Degrees: d_0 = 2, d_1 = 1, d_2 = 1, d_3 = 2, d_4 = 0
    # D^{-1/2} = [1/sqrt(2), 1, 1, 1/sqrt(2), 0]
    #
    # Normalized weights:
    # (u0, i0) -> A_0,2 = 1 / (sqrt(2) * sqrt(1)) = 1 / sqrt(2) ≈ 0.70710678
    # (u0, i1) -> A_0,3 = 1 / (sqrt(2) * sqrt(2)) = 1 / 2 = 0.5
    # (u1, i1) -> A_1,3 = 1 / (sqrt(1) * sqrt(2)) = 1 / sqrt(2) ≈ 0.70710678
    # Isolated node i2 -> row 4 and col 4 are strictly all 0.0
    norm_adj = build_normalized_adjacency(num_users, num_items, train_pairs)
    dense_norm = norm_adj.toarray()

    # Symmetry check
    np.testing.assert_allclose(dense_norm, dense_norm.T, rtol=1e-6)

    # Check non-zero values
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    assert np.isclose(dense_norm[0, 2], inv_sqrt2)
    assert np.isclose(dense_norm[2, 0], inv_sqrt2)
    assert np.isclose(dense_norm[0, 3], 0.5)
    assert np.isclose(dense_norm[3, 0], 0.5)
    assert np.isclose(dense_norm[1, 3], inv_sqrt2)
    assert np.isclose(dense_norm[3, 1], inv_sqrt2)

    # Isolated node check: row 4 and col 4 must be zero, no NaNs or Infs
    assert not np.isnan(dense_norm).any()
    assert not np.isinf(dense_norm).any()
    np.testing.assert_array_equal(dense_norm[4, :], 0.0)
    np.testing.assert_array_equal(dense_norm[:, 4], 0.0)


def test_torch_sparse_equivalence():
    """Verify conversion to PyTorch sparse tensor preserves values and dimensions."""
    num_users = 3
    num_items = 4
    train_pairs = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 3],
    ], dtype=np.int64)

    sp_norm = build_normalized_adjacency(num_users, num_items, train_pairs)
    torch_norm = scipy_coo_to_torch_sparse(sp_norm)

    assert torch_norm.is_sparse
    assert torch_norm.is_coalesced()
    assert torch_norm.shape == (num_users + num_items, num_users + num_items)

    dense_sp = sp_norm.toarray()
    dense_torch = torch_norm.to_dense().numpy()
    np.testing.assert_allclose(dense_sp, dense_torch, rtol=1e-6)
