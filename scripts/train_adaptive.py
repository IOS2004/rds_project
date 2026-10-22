"""Training script for AdaptiveLightGCN with learnable layer aggregation weights."""

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
from src.models.adaptive_lightgcn import AdaptiveLightGCN
from src.trainer import Trainer, TrainConfig
from scripts.train_baseline import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Train AdaptiveLightGCN model.")
    parser.add_argument("--data_dir", type=str, default="data/ml-100k", help="Path to dataset directory.")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension d.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of propagation layers K.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument("--reg_weight", type=float, default=1e-4, help="L2 regularization lambda.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--eval_every", type=int, default=5, help="Validation interval in epochs.")
    parser.add_argument("--early_stop_patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--save_dir", type=str, default="experiments/adaptive", help="Directory to save checkpoints.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading data from {args.data_dir}...", flush=True)
    data = load_dataset(args.data_dir)
    print(data.get_stats(), flush=True)

    print("Building normalized adjacency matrix...", flush=True)
    norm_adj = build_torch_normalized_adjacency(data.num_users, data.num_items, data.train_pairs)

    print(f"Initializing AdaptiveLightGCN (dim={args.embedding_dim}, layers={args.num_layers})...", flush=True)
    model = AdaptiveLightGCN(
        num_users=data.num_users,
        num_items=data.num_items,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
    )

    init_weights = model.get_layer_weights().detach().cpu().numpy()
    print(f"Initial uniform layer weights alpha: {init_weights.round(4).tolist()}", flush=True)

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

    # Log learned layer weights
    final_weights = model.get_layer_weights().detach().cpu().numpy()
    final_logits = model.layer_logits.detach().cpu().numpy()
    print("\nLearned Layer Combination Parameters:")
    for k, (w, logit) in enumerate(zip(final_weights, final_logits)):
        print(f"  Layer {k}: weight alpha_{k} = {w:.4f} (logit theta_{k} = {logit:.4f})")

    # Save weights info to summary
    summary["learned_weights"] = final_weights.tolist()
    summary["learned_logits"] = final_logits.tolist()
    summary["initial_weights"] = init_weights.tolist()

    import json
    with open(os.path.join(args.save_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    history_df = pd.DataFrame(trainer.history)
    history_df.to_csv(os.path.join(args.save_dir, "training_history.csv"), index=False)
    print(f"Saved training history to {args.save_dir}/training_history.csv", flush=True)


if __name__ == "__main__":
    main()
