"""Training script for Baseline LightGCN."""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import pandas as pd
import numpy as np

from src.dataset import process_interaction_df, InteractionData
from src.graph import build_torch_normalized_adjacency
from src.models.lightgcn import LightGCN
from src.trainer import Trainer, TrainConfig


def load_dataset(data_dir: str) -> InteractionData:
    """Load train, val, and test splits from data_dir."""
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            f"Dataset files not found in {data_dir}. "
            f"Please run 'python data/download_data.py --dataset ml-100k' first."
        )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    all_users = set(train_df["user_id"]).union(set(val_df["user_id"])).union(set(test_df["user_id"]))
    all_items = set(train_df["item_id"]).union(set(val_df["item_id"])).union(set(test_df["item_id"]))

    num_users = max(all_users) + 1 if all_users else 0
    num_items = max(all_items) + 1 if all_items else 0

    return InteractionData(
        num_users=num_users,
        num_items=num_items,
        train_pairs=train_df[["user_id", "item_id"]].to_numpy(dtype=np.int64),
        val_pairs=val_df[["user_id", "item_id"]].to_numpy(dtype=np.int64),
        test_pairs=test_df[["user_id", "item_id"]].to_numpy(dtype=np.int64),
    )


def main():
    parser = argparse.ArgumentParser(description="Train Baseline LightGCN model.")
    parser.add_argument("--data_dir", type=str, default="data/ml-100k", help="Path to dataset directory.")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension d.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of propagation layers K.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument("--reg_weight", type=float, default=1e-4, help="L2 regularization lambda.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--eval_every", type=int, default=5, help="Validation interval in epochs.")
    parser.add_argument("--early_stop_patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--save_dir", type=str, default="experiments/baseline", help="Directory to save checkpoints.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading data from {args.data_dir}...")
    data = load_dataset(args.data_dir)
    stats = data.get_stats()
    print(stats)

    print("Building normalized adjacency matrix \\tilde{A} = D^{-1/2} A D^{-1/2}...")
    norm_adj = build_torch_normalized_adjacency(data.num_users, data.num_items, data.train_pairs)

    print(f"Initializing Baseline LightGCN (dim={args.embedding_dim}, layers={args.num_layers})...")
    model = LightGCN(
        num_users=data.num_users,
        num_items=data.num_items,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
    )

    config = TrainConfig(
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        reg_weight=args.reg_weight,
        batch_size=args.batch_size,
        epochs=args.epochs,
        eval_every=args.eval_every,
        early_stop_patience=args.early_stop_patience,
        early_stop_metric="Recall@20",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        data=data,
        norm_adj=norm_adj,
        config=config,
        save_dir=args.save_dir,
    )

    summary = trainer.train()

    # Save history dataframe
    history_df = pd.DataFrame(trainer.history)
    history_df.to_csv(os.path.join(args.save_dir, "training_history.csv"), index=False)
    print(f"Saved training history to {args.save_dir}/training_history.csv")


if __name__ == "__main__":
    main()
