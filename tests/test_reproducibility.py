"""Unit tests for training reproducibility with fixed random seeds."""

import numpy as np
import torch
import pytest

from src.dataset import process_interaction_df
from src.graph import build_torch_normalized_adjacency
from src.models.lightgcn import LightGCN
from src.trainer import Trainer, TrainConfig


def run_mini_training(seed: int):
    # Fix all seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Hand-crafted small interaction dataset
    users = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    items = [0, 1, 1, 2, 2, 3, 3, 4, 0, 4]
    import pandas as pd
    df = pd.DataFrame({"user_id": users, "item_id": items})
    data = process_interaction_df(df, val_ratio=0.2, test_ratio=0.2, seed=seed)

    norm_adj = build_torch_normalized_adjacency(data.num_users, data.num_items, data.train_pairs)

    model = LightGCN(data.num_users, data.num_items, embedding_dim=8, num_layers=2)

    config = TrainConfig(
        embedding_dim=8,
        num_layers=2,
        learning_rate=0.01,
        reg_weight=1e-4,
        batch_size=4,
        epochs=3,
        eval_every=1,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        data=data,
        norm_adj=norm_adj,
        config=config,
        save_dir=f"experiments/test_repro_{seed}",
    )

    loss_history = []
    for _ in range(config.epochs):
        ep_loss = trainer.train_epoch()
        loss_history.append(ep_loss["train_loss"])

    final_emb = model.user_embedding.weight.detach().cpu().numpy()
    return loss_history, final_emb


def test_seed_reproducibility():
    """Verify that identical random seeds yield identical training losses and embeddings."""
    loss1, emb1 = run_mini_training(seed=42)
    loss2, emb2 = run_mini_training(seed=42)
    loss3, emb3 = run_mini_training(seed=99)

    # Identical seeds must match exactly
    np.testing.assert_allclose(loss1, loss2, rtol=1e-5)
    np.testing.assert_allclose(emb1, emb2, rtol=1e-5)

    # Different seeds must diverge
    assert not np.allclose(loss1, loss3, rtol=1e-3)
    assert not np.allclose(emb1, emb3, rtol=1e-3)
