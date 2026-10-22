"""Unit tests for top-N ranking metrics and training item exclusion."""

import numpy as np
import torch
import pytest

from src.metrics import compute_recall_at_k, compute_ndcg_at_k, evaluate_model
from src.models.lightgcn import LightGCN
from src.graph import build_torch_normalized_adjacency


def test_hand_calculated_recall_and_ndcg():
    """Verify Recall@K and NDCG@K against exact analytical values."""
    # User ground truth items: {2, 4} (2 items)
    ground_truth = {2, 4}

    # Recommended items ordered by predicted ranking score
    recommended = [1, 2, 3, 5, 4]

    # At K=2: top items = [1, 2]
    # Hits = {2} -> Recall@2 = 1 / 2 = 0.5
    recall_2 = compute_recall_at_k(recommended, ground_truth, k=2)
    assert pytest.approx(recall_2, rel=1e-5) == 0.5

    # DCG@2: item 2 at rank 2 (index 1) -> 1 / log2(2 + 1) = 1 / log2(3)
    # IDCG@2: 1 / log2(2) + 1 / log2(3) = 1 + 1 / log2(3)
    log2_3 = np.log2(3.0)
    expected_dcg_2 = 1.0 / log2_3
    expected_idcg_2 = 1.0 + (1.0 / log2_3)
    expected_ndcg_2 = expected_dcg_2 / expected_idcg_2
    ndcg_2 = compute_ndcg_at_k(recommended, ground_truth, k=2)
    assert pytest.approx(ndcg_2, rel=1e-5) == expected_ndcg_2

    # At K=5: top items = [1, 2, 3, 5, 4]
    # Hits = {2, 4} -> Recall@5 = 2 / 2 = 1.0
    recall_5 = compute_recall_at_k(recommended, ground_truth, k=5)
    assert pytest.approx(recall_5, rel=1e-5) == 1.0

    # DCG@5: item 2 at rank 2, item 4 at rank 5 -> 1/log2(3) + 1/log2(6)
    # IDCG@5: min(len(gt), 5) = 2 -> IDCG@5 = IDCG@2
    expected_dcg_5 = (1.0 / log2_3) + (1.0 / np.log2(6.0))
    expected_ndcg_5 = expected_dcg_5 / expected_idcg_2
    ndcg_5 = compute_ndcg_at_k(recommended, ground_truth, k=5)
    assert pytest.approx(ndcg_5, rel=1e-5) == expected_ndcg_5


def test_training_item_exclusion_during_evaluation():
    """Verify that training items are strictly masked and never recommended."""
    num_users = 1
    num_items = 4

    # User 0 interacted with item 0 in training
    train_pairs = np.array([[0, 0]], dtype=np.int64)
    train_positives = {0: {0}}

    # User 0 interacted with item 1 in test set
    test_positives = {0: {1}}

    norm_adj = build_torch_normalized_adjacency(num_users, num_items, train_pairs)
    model = LightGCN(num_users, num_items, embedding_dim=2, num_layers=1)

    # Force item 0 (the training item) to have the highest possible dot product
    with torch.no_grad():
        model.user_embedding.weight.copy_(torch.tensor([[10.0, 10.0]]))
        model.item_embedding.weight.copy_(torch.tensor([
            [10.0, 10.0],  # item 0 (training item) - highest raw score
            [2.0, 2.0],    # item 1 (test item)
            [1.0, 1.0],    # item 2
            [0.5, 0.5],    # item 3
        ]))

    # Without exclusion, item 0 would rank #1
    # With exclusion, item 0 must be masked to -inf, so item 1 ranks #1
    metrics = evaluate_model(
        model=model,
        norm_adj=norm_adj,
        eval_user_positives=test_positives,
        train_user_positives=train_positives,
        k_list=(1, 2),
    )

    # Item 1 is at rank 1 -> Recall@1 must be 1.0 and NDCG@1 must be 1.0
    assert metrics["Recall@1"] == 1.0
    assert metrics["NDCG@1"] == 1.0
