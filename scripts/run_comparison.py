"""Experimental comparison runner: Baseline LightGCN vs AdaptiveLightGCN vs LightGCN-CCL.

Performs controlled multi-seed evaluation:
1. Trains Baseline (BPR), Adaptive (BPR), and LightGCN-CCL (SimpleX CCL) on identical splits and seeds.
2. Computes mean and standard deviation for Recall@10, Recall@20, NDCG@10, NDCG@20.
3. Records learned aggregation weights across seeds for AdaptiveLightGCN.
4. Generates comprehensive comparison tables (CSV, JSON) and publication-quality plots.
"""

import argparse
import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.dataset import InteractionData
from src.graph import build_torch_normalized_adjacency
from src.models.lightgcn import LightGCN
from src.models.adaptive_lightgcn import AdaptiveLightGCN
from src.models.lightgcn_ccl import LightGCNCCL
from src.trainer import Trainer, TrainConfig
from scripts.train_baseline import load_dataset


def run_experiment_for_model(
    model_cls,
    model_name: str,
    data: InteractionData,
    norm_adj: torch.Tensor,
    seed: int,
    epochs: int,
    embedding_dim: int,
    num_layers: int,
    lr: float,
    reg_weight: float,
    batch_size: int,
    save_dir: str,
    loss_type: str = "bpr",
    ccl_margin: float = 0.8,
    ccl_negative_weight: float = 10.0,
    num_negatives: int = 1,
    score_type: str = "dot",
):
    """Train and evaluate a specific model variant for a given seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if model_cls == LightGCNCCL:
        model = model_cls(
            num_users=data.num_users,
            num_items=data.num_items,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            score_type=score_type,
        )
    else:
        model = model_cls(
            num_users=data.num_users,
            num_items=data.num_items,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
        )

    config = TrainConfig(
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        learning_rate=lr,
        reg_weight=reg_weight,
        batch_size=batch_size,
        epochs=epochs,
        eval_every=5,
        early_stop_patience=6,
        early_stop_metric="Recall@20",
        seed=seed,
        loss_type=loss_type,
        ccl_margin=ccl_margin,
        ccl_negative_weight=ccl_negative_weight,
        num_negatives=num_negatives,
        score_type=score_type,
    )

    trainer = Trainer(
        model=model,
        data=data,
        norm_adj=norm_adj,
        config=config,
        save_dir=save_dir,
    )

    summary = trainer.train()

    learned_weights = None
    if hasattr(model, "get_layer_weights"):
        learned_weights = model.get_layer_weights().detach().cpu().numpy().tolist()

    return summary["test_metrics"], trainer.history, learned_weights


def plot_comparison(
    results_df: pd.DataFrame,
    learned_weights_list: list,
    save_dir: str,
    num_negatives: int = 100,
):
    """Generate 3-model comparison bar charts and learned weight plots."""
    os.makedirs(save_dir, exist_ok=True)

    metrics = ["Recall@10", "Recall@20", "NDCG@10", "NDCG@20"]
    x = np.arange(len(metrics))

    models = results_df.index.tolist()
    has_three_models = all(m in models for m in ["Baseline", "Adaptive", "LightGCN-CCL"])

    fig, ax = plt.subplots(figsize=(10, 5.5))

    if has_three_models:
        width = 0.25
        base_means = [results_df.loc["Baseline", f"{m}_mean"] for m in metrics]
        base_stds = [results_df.loc["Baseline", f"{m}_std"] if not np.isnan(results_df.loc["Baseline", f"{m}_std"]) else 0.0 for m in metrics]

        adapt_means = [results_df.loc["Adaptive", f"{m}_mean"] for m in metrics]
        adapt_stds = [results_df.loc["Adaptive", f"{m}_std"] if not np.isnan(results_df.loc["Adaptive", f"{m}_std"]) else 0.0 for m in metrics]

        ccl_means = [results_df.loc["LightGCN-CCL", f"{m}_mean"] for m in metrics]
        ccl_stds = [results_df.loc["LightGCN-CCL", f"{m}_std"] if not np.isnan(results_df.loc["LightGCN-CCL", f"{m}_std"]) else 0.0 for m in metrics]

        ax.bar(x - width, base_means, width, yerr=base_stds, label="Baseline LightGCN (BPR)", capsize=4, color="#1f77b4")
        ax.bar(x, adapt_means, width, yerr=adapt_stds, label="AdaptiveLightGCN (BPR)", capsize=4, color="#ff7f0e")
        ax.bar(x + width, ccl_means, width, yerr=ccl_stds, label=f"LightGCN-CCL (CCL, |N|={num_negatives})", capsize=4, color="#2ca02c")
    else:
        width = 0.35
        for idx, m in enumerate(models):
            means = [results_df.loc[m, f"{metric}_mean"] for metric in metrics]
            stds = [results_df.loc[m, f"{metric}_std"] if not np.isnan(results_df.loc[m, f"{metric}_std"]) else 0.0 for metric in metrics]
            ax.bar(x + (idx - len(models) / 2.0 + 0.5) * width, means, width, yerr=stds, label=m, capsize=4)

    ax.set_ylabel("Metric Score", fontsize=12)
    ax.set_title("LightGCN Variants Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plot_path = os.path.join(save_dir, "metric_comparison.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved metric comparison plot to {plot_path}", flush=True)

    # Learned Weights Visualization (for Adaptive)
    if learned_weights_list:
        weights_arr = np.array(learned_weights_list)  # (num_seeds, num_layers + 1)
        mean_weights = weights_arr.mean(axis=0)
        std_weights = weights_arr.std(axis=0)
        num_layers = len(mean_weights)
        uniform_val = 1.0 / num_layers

        fig, ax = plt.subplots(figsize=(7, 4.5))
        layers = [f"Layer {k}" for k in range(num_layers)]
        ax.bar(layers, mean_weights, yerr=std_weights, color="#ff7f0e", capsize=5, label="Learned Mean Weight")
        ax.axhline(uniform_val, color="red", linestyle="--", linewidth=1.5, label=f"Uniform Weight ({uniform_val:.3f})")

        ax.set_ylabel("Aggregation Weight (alpha_k)", fontsize=12)
        ax.set_title("AdaptiveLightGCN Learned Layer Combination Weights", fontsize=13, fontweight="bold")
        ax.set_ylim(0, max(mean_weights.max() * 1.3, uniform_val * 1.5))
        ax.legend(fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        weights_plot_path = os.path.join(save_dir, "learned_weights.png")
        fig.tight_layout()
        fig.savefig(weights_plot_path, dpi=300)
        plt.close(fig)
        print(f"Saved learned weights plot to {weights_plot_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run controlled comparison among LightGCN baseline, AdaptiveLightGCN, and LightGCN-CCL."
    )
    parser.add_argument("--data_dir", type=str, default="data/ml-100k", help="Path to processed dataset.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456], help="Random seeds to evaluate.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per run.")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension d.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of propagation layers K.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--reg_weight", type=float, default=1e-4, help="L2 regularization.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size.")
    parser.add_argument("--save_dir", type=str, default="experiments/comparison", help="Output directory.")
    # CCL Hyperparameters
    parser.add_argument(
        "--ccl_margin",
        type=float,
        default=0.8,
        help="Negative similarity margin m for CCL (default: 0.8).",
    )
    parser.add_argument(
        "--ccl_negative_weight",
        type=float,
        default=10.0,
        help="Negative loss weight w for CCL (default: 10.0).",
    )
    parser.add_argument(
        "--ccl_num_negatives",
        type=int,
        default=100,
        help="Number of sampled negatives per positive interaction for CCL (default: 100).",
    )
    parser.add_argument(
        "--ccl_score_type",
        type=str,
        default="cosine",
        choices=["cosine", "dot"],
        help="Scoring metric for LightGCN-CCL ('cosine' or 'dot', default: 'cosine').",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Loading data from {args.data_dir}...", flush=True)
    data = load_dataset(args.data_dir)
    norm_adj = build_torch_normalized_adjacency(data.num_users, data.num_items, data.train_pairs)

    all_results = []
    adaptive_weights = []

    for seed in args.seeds:
        print(f"\n================ Running Seed {seed} ================", flush=True)

        # 1. Baseline LightGCN (BPR)
        base_dir = os.path.join(args.save_dir, f"baseline_seed_{seed}")
        print(f"--- Training Baseline LightGCN (Seed {seed}) ---", flush=True)
        base_test_metrics, _, _ = run_experiment_for_model(
            model_cls=LightGCN,
            model_name="Baseline",
            data=data,
            norm_adj=norm_adj,
            seed=seed,
            epochs=args.epochs,
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
            lr=args.learning_rate,
            reg_weight=args.reg_weight,
            batch_size=args.batch_size,
            save_dir=base_dir,
            loss_type="bpr",
            num_negatives=1,
            score_type="dot",
        )
        base_row = {"model": "Baseline", "seed": seed} | base_test_metrics
        all_results.append(base_row)

        # 2. AdaptiveLightGCN (BPR)
        adapt_dir = os.path.join(args.save_dir, f"adaptive_seed_{seed}")
        print(f"\n--- Training AdaptiveLightGCN (Seed {seed}) ---", flush=True)
        adapt_test_metrics, _, weights = run_experiment_for_model(
            model_cls=AdaptiveLightGCN,
            model_name="Adaptive",
            data=data,
            norm_adj=norm_adj,
            seed=seed,
            epochs=args.epochs,
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
            lr=args.learning_rate,
            reg_weight=args.reg_weight,
            batch_size=args.batch_size,
            save_dir=adapt_dir,
            loss_type="bpr",
            num_negatives=1,
            score_type="dot",
        )
        adapt_row = {"model": "Adaptive", "seed": seed} | adapt_test_metrics
        all_results.append(adapt_row)
        if weights:
            adaptive_weights.append(weights)

        # 3. LightGCN-CCL (SimpleX CCL)
        ccl_dir = os.path.join(args.save_dir, f"lightgcn_ccl_seed_{seed}")
        print(f"\n--- Training LightGCN-CCL (Seed {seed}, |N|={args.ccl_num_negatives}) ---", flush=True)
        ccl_test_metrics, _, _ = run_experiment_for_model(
            model_cls=LightGCNCCL,
            model_name="LightGCN-CCL",
            data=data,
            norm_adj=norm_adj,
            seed=seed,
            epochs=args.epochs,
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
            lr=args.learning_rate,
            reg_weight=args.reg_weight,
            batch_size=args.batch_size,
            save_dir=ccl_dir,
            loss_type="ccl",
            ccl_margin=args.ccl_margin,
            ccl_negative_weight=args.ccl_negative_weight,
            num_negatives=args.ccl_num_negatives,
            score_type=args.ccl_score_type,
        )
        ccl_row = {"model": "LightGCN-CCL", "seed": seed} | ccl_test_metrics
        all_results.append(ccl_row)

    # Compile results
    raw_df = pd.DataFrame(all_results)
    raw_csv_path = os.path.join(args.save_dir, "raw_seed_metrics.csv")
    raw_df.to_csv(raw_csv_path, index=False)

    agg_df = raw_df.groupby("model")[["Recall@10", "Recall@20", "NDCG@10", "NDCG@20"]].agg(["mean", "std"])

    # Flatten MultiIndex columns to single string names: e.g. Recall@10_mean
    summary_df = pd.DataFrame(index=agg_df.index)
    for col in ["Recall@10", "Recall@20", "NDCG@10", "NDCG@20"]:
        summary_df[f"{col}_mean"] = agg_df[(col, "mean")]
        summary_df[f"{col}_std"] = agg_df[(col, "std")]

    summary_csv_path = os.path.join(args.save_dir, "comparison_summary.csv")
    summary_df.to_csv(summary_csv_path)

    print("\n================ FINAL COMPARISON SUMMARY ================", flush=True)
    print(summary_df.round(4), flush=True)

    # Generate plots
    plot_comparison(summary_df, adaptive_weights, args.save_dir, num_negatives=args.ccl_num_negatives)

    # Save complete JSON
    experiment_config = {
        "dataset": args.data_dir,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "embedding_dim": args.embedding_dim,
        "num_layers": args.num_layers,
        "learning_rate": args.learning_rate,
        "reg_weight": args.reg_weight,
        "batch_size": args.batch_size,
        "ccl_margin": args.ccl_margin,
        "ccl_negative_weight": args.ccl_negative_weight,
        "ccl_num_negatives": args.ccl_num_negatives,
        "ccl_score_type": args.ccl_score_type,
        "candidate_ranking_protocol": "Full catalog unobserved items; observed training items masked to -inf",
        "models_evaluated": ["Baseline (BPR)", "Adaptive (BPR)", "LightGCN-CCL (CCL)"],
    }
    summary_dict = {
        "config": experiment_config,
        "raw_results": all_results,
        "summary": summary_df.to_dict(orient="index"),
        "learned_weights_per_seed": adaptive_weights,
    }
    with open(os.path.join(args.save_dir, "comparison_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    print(f"\nSaved full results to {args.save_dir}", flush=True)


if __name__ == "__main__":
    main()
