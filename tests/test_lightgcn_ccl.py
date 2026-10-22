"""Unit tests for LightGCNCCL model and end-to-end training and evaluation pipeline."""

import pytest
import numpy as np
import torch

from src.dataset import InteractionData
from src.graph import build_torch_normalized_adjacency
from src.models.lightgcn_ccl import LightGCNCCL
from src.trainer import Trainer, TrainConfig
from src.metrics import evaluate_model


@pytest.fixture
def toy_recommendation_setup():
    """Create a toy recommendation dataset and normalized adjacency matrix."""
    num_users = 4
    num_items = 8

    # Training pairs
    train_pairs = np.array([
        [0, 0], [0, 1],
        [1, 1], [1, 2],
        [2, 3], [2, 4],
        [3, 5], [3, 6],
    ], dtype=np.int64)

    # Validation pairs
    val_pairs = np.array([
        [0, 2],
        [1, 3],
        [2, 5],
    ], dtype=np.int64)

    # Test pairs
    test_pairs = np.array([
        [0, 3],
        [1, 4],
        [3, 7],
    ], dtype=np.int64)

    data = InteractionData(
        num_users=num_users,
        num_items=num_items,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
    )

    norm_adj = build_torch_normalized_adjacency(num_users, num_items, train_pairs)
    return data, norm_adj


def test_lightgcn_ccl_model_forward_and_prediction(toy_recommendation_setup):
    """Verify LightGCNCCL forward pass and predict methods."""
    data, norm_adj = toy_recommendation_setup

    model = LightGCNCCL(
        num_users=data.num_users,
        num_items=data.num_items,
        embedding_dim=8,
        num_layers=2,
        score_type="cosine",
    )

    users = torch.tensor([0, 1], dtype=torch.long)
    pos_items = torch.tensor([0, 1], dtype=torch.long)
    neg_items = torch.tensor([[2, 3, 4], [0, 3, 5]], dtype=torch.long)

    u_final, pos_final, neg_final, u0, pos0, neg0 = model(users, pos_items, neg_items, norm_adj)

    assert u_final.shape == (2, 8)
    assert pos_final.shape == (2, 8)
    assert neg_final.shape == (2, 3, 8)
    assert u0.shape == (2, 8)
    assert pos0.shape == (2, 8)
    assert neg0.shape == (2, 3, 8)

    # Test predict_all_for_users
    all_scores = model.predict_all_for_users(users, norm_adj)
    assert all_scores.shape == (2, data.num_items)
    # Cosine similarities should be in [-1, 1] within floating-point tolerance
    assert (all_scores >= -1.0001).all() and (all_scores <= 1.0001).all()

    # Test predict for explicit pairs
    pair_scores = model.predict(users, pos_items, norm_adj)
    assert pair_scores.shape == (2,)
    assert (pair_scores >= -1.0001).all() and (pair_scores <= 1.0001).all()


def test_lightgcn_ccl_training_step(toy_recommendation_setup, tmp_path):
    """Verify that LightGCNCCL executes training epochs and updates weights."""
    data, norm_adj = toy_recommendation_setup

    model = LightGCNCCL(
        num_users=data.num_users,
        num_items=data.num_items,
        embedding_dim=8,
        num_layers=2,
        score_type="cosine",
    )

    initial_user_weights = model.user_embedding.weight.clone().detach()

    config = TrainConfig(
        embedding_dim=8,
        num_layers=2,
        learning_rate=0.01,
        reg_weight=1e-4,
        batch_size=4,
        epochs=3,
        eval_every=1,
        seed=42,
        loss_type="ccl",
        ccl_margin=0.5,
        ccl_negative_weight=5.0,
        num_negatives=3,
        score_type="cosine",
    )

    trainer = Trainer(
        model=model,
        data=data,
        norm_adj=norm_adj,
        config=config,
        save_dir=str(tmp_path),
    )

    epoch_res = trainer.train_epoch()
    assert "train_loss" in epoch_res
    assert "ccl_loss" in epoch_res
    assert "reg_loss" in epoch_res
    assert np.isfinite(epoch_res["train_loss"])
    assert epoch_res["train_loss"] > 0.0

    # Verify that initial embeddings were updated by gradient descent
    final_user_weights = model.user_embedding.weight.detach()
    assert not torch.allclose(initial_user_weights, final_user_weights)


def test_lightgcn_ccl_evaluation_metrics_finite(toy_recommendation_setup):
    """Verify evaluate_model returns finite Recall and NDCG scores with cosine scoring."""
    data, norm_adj = toy_recommendation_setup

    model = LightGCNCCL(
        num_users=data.num_users,
        num_items=data.num_items,
        embedding_dim=8,
        num_layers=2,
        score_type="cosine",
    )

    metrics = evaluate_model(
        model=model,
        norm_adj=norm_adj,
        eval_user_positives=data.val_user_positives,
        train_user_positives=data.train_user_positives,
        k_list=(2, 4),
        score_type="cosine",
    )

    assert "Recall@2" in metrics
    assert "Recall@4" in metrics
    assert "NDCG@2" in metrics
    assert "NDCG@4" in metrics

    for k, val in metrics.items():
        assert np.isfinite(val), f"Metric {k} is not finite: {val}"
        assert 0.0 <= val <= 1.0, f"Metric {k} is out of [0, 1] range: {val}"
