"""Training script for LightGCN-CCL with Cosine Contrastive Loss.

Faithfully implements the LightGCN-CCL training variant based on:
"SimpleX: A Simple and Strong Baseline for Collaborative Filtering"
Kelong Mao et al., CIKM 2021 (arXiv:2109.12613).
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import pandas as pd
import numpy as np

from src.dataset import InteractionData
from src.graph import build_torch_normalized_adjacency
from src.models.lightgcn_ccl import LightGCNCCL
from src.trainer import Trainer, TrainConfig
from scripts.train_baseline import load_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Train LightGCN-CCL model with Cosine Contrastive Loss (SimpleX paper)."
    )
    parser.add_argument("--data_dir", type=str, default="data/ml-100k", help="Path to dataset directory.")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension d.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of propagation layers K.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument("--reg_weight", type=float, default=1e-4, help="L2 regularization lambda.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--eval_every", type=int, default=5, help="Validation interval in epochs.")
    parser.add_argument("--early_stop_patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--save_dir", type=str, default="experiments/lightgcn_ccl", help="Directory to save checkpoints.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    # CCL-specific hyperparameters
    parser.add_argument(
        "--ccl_margin",
        type=float,
        default=0.8,
        help="Negative similarity margin m for CCL (default: 0.8). Negatives with sim <= m incur 0 penalty.",
    )
    parser.add_argument(
        "--ccl_negative_weight",
        type=float,
        default=10.0,
        help="Negative loss scaling weight w in CCL (default: 10.0).",
    )
    parser.add_argument(
        "--num_negatives",
        type=int,
        default=100,
        help="Number of negative samples per positive interaction |N| (default: 100).",
    )
    parser.add_argument(
        "--score_type",
        type=str,
        default="cosine",
        choices=["cosine", "dot"],
        help="Scoring metric for evaluation and ranking ('cosine' or 'dot', default: 'cosine').",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="ccl",
        choices=["ccl", "cosine_softmax"],
        help="Loss objective: 'ccl' (Hinge Margin Cosine Contrastive Loss) or 'cosine_softmax' (Sampled InfoNCE).",
    )
    parser.add_argument(
        "--softmax_temperature",
        type=float,
        default=0.15,
        help="Temperature parameter tau for cosine_softmax loss (default: 0.15).",
    )
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading data from {args.data_dir}...", flush=True)
    data = load_dataset(args.data_dir)
    stats = data.get_stats()
    print(stats, flush=True)

    print("Building normalized adjacency matrix \\tilde{A} = D^{-1/2} A D^{-1/2}...", flush=True)
    norm_adj = build_torch_normalized_adjacency(data.num_users, data.num_items, data.train_pairs)

    print(
        f"Initializing LightGCNCCL (dim={args.embedding_dim}, layers={args.num_layers}, "
        f"score_type={args.score_type})...",
        flush=True,
    )
    model = LightGCNCCL(
        num_users=data.num_users,
        num_items=data.num_items,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        score_type=args.score_type,
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
        loss_type=args.loss_type,
        ccl_margin=args.ccl_margin,
        ccl_negative_weight=args.ccl_negative_weight,
        softmax_temperature=args.softmax_temperature,
        num_negatives=args.num_negatives,
        score_type=args.score_type,
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
    print(f"Saved training history to {args.save_dir}/training_history.csv", flush=True)


if __name__ == "__main__":
    main()
