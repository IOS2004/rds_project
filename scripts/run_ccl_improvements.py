"""Benchmark script comparing Margin Calibration and Cosine Softmax Contrastive Loss (Sampled InfoNCE).

Both methods use the exact same LightGCNCCL model architecture:
- 3 layers LightGCN propagation
- Uniform layer aggregation (alpha = 1 / (K + 1))
- Cosine similarity scoring for top-K ranking
- 50 sampled unobserved negative items

No new model architecture is created.
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.train_baseline import load_dataset
from src.graph import build_torch_normalized_adjacency
from src.models.lightgcn_ccl import LightGCNCCL
from src.trainer import Trainer, TrainConfig


def run_experiment(
    exp_name: str,
    loss_type: str,
    ccl_margin: float,
    softmax_temperature: float,
    data,
    norm_adj,
    args,
    save_dir: str,
) -> dict:
    exp_dir = os.path.join(save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    summary_path = os.path.join(exp_dir, "train_summary.json")

    if os.path.exists(summary_path):
        print(f"Loading cached results from {summary_path}...", flush=True)
        with open(summary_path, "r") as f:
            summary = json.load(f)
        metrics = summary["test_metrics"]
        return {
            "experiment": exp_name,
            "loss_type": loss_type,
            "margin": ccl_margin if loss_type == "ccl" else None,
            "temperature": softmax_temperature if loss_type == "cosine_softmax" else None,
            "Recall@10": metrics["Recall@10"],
            "Recall@20": metrics["Recall@20"],
            "NDCG@10": metrics["NDCG@10"],
            "NDCG@20": metrics["NDCG@20"],
        }

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = LightGCNCCL(
        num_users=data.num_users,
        num_items=data.num_items,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        score_type="cosine",
    )

    config = TrainConfig(
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        reg_weight=args.reg_weight,
        batch_size=args.batch_size,
        epochs=args.epochs,
        eval_every=5,
        early_stop_patience=5,
        early_stop_metric="Recall@20",
        seed=args.seed,
        loss_type=loss_type,
        ccl_margin=ccl_margin,
        ccl_negative_weight=args.ccl_negative_weight,
        softmax_temperature=softmax_temperature,
        num_negatives=args.num_negatives,
        score_type="cosine",
    )

    exp_dir = os.path.join(save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    trainer = Trainer(
        model=model,
        data=data,
        norm_adj=norm_adj,
        config=config,
        save_dir=exp_dir,
    )

    summary = trainer.train()
    metrics = summary["test_metrics"]

    return {
        "experiment": exp_name,
        "loss_type": loss_type,
        "margin": ccl_margin if loss_type == "ccl" else None,
        "temperature": softmax_temperature if loss_type == "cosine_softmax" else None,
        "Recall@10": metrics["Recall@10"],
        "Recall@20": metrics["Recall@20"],
        "NDCG@10": metrics["NDCG@10"],
        "NDCG@20": metrics["NDCG@20"],
    }


def main():
    parser = argparse.ArgumentParser(description="Run CCL Improvement Experiments")
    parser.add_argument("--data_dir", type=str, default="data/ml-100k")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--reg_weight", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_negatives", type=int, default=50)
    parser.add_argument("--ccl_negative_weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="experiments/ccl_improvements")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Loading data from {args.data_dir}...", flush=True)
    data = load_dataset(args.data_dir)
    norm_adj = build_torch_normalized_adjacency(data.num_users, data.num_items, data.train_pairs)

    experiments = [
        # Baseline Hinge CCL
        ("Baseline_CCL_m0.8", "ccl", 0.8, 0.1),
        # Margin Calibration
        ("Margin_Calibration_m0.6", "ccl", 0.6, 0.1),
        ("Margin_Calibration_m0.4", "ccl", 0.4, 0.1),
        # Cosine Softmax Contrastive Loss (Sampled InfoNCE)
        ("Softmax_Contrastive_tau0.15", "cosine_softmax", 0.8, 0.15),
        ("Softmax_Contrastive_tau0.10", "cosine_softmax", 0.8, 0.10),
        ("Softmax_Contrastive_tau0.07", "cosine_softmax", 0.8, 0.07),
    ]

    results = []
    for exp_name, loss_type, margin, tau in experiments:
        print(f"\n================ Running: {exp_name} ================", flush=True)
        res = run_experiment(exp_name, loss_type, margin, tau, data, norm_adj, args, args.save_dir)
        results.append(res)
        print(
            f"Result for {exp_name}: Recall@20 = {res['Recall@20']:.4f}, NDCG@20 = {res['NDCG@20']:.4f}",
            flush=True,
        )

    df = pd.DataFrame(results)
    csv_path = os.path.join(args.save_dir, "ccl_improvements_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary to {csv_path}", flush=True)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
