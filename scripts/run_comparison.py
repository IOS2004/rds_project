"""Experimental comparison runner for all LightGCN variants on MovieLens-100K.

Evaluates 5 distinct model/loss configurations:
1. Baseline LightGCN (Linear bipartite propagation + BPR loss + uniform layer combination)
2. AdaptiveLightGCN (Trainable softmax layer weights + BPR loss)
3. LightGCN-CCL (m=0.8) (Original SimpleX Cosine Contrastive Loss, margin m=0.8, |N|=50)
4. LightGCN-CCL (m=0.6) (Calibrated Hinge Margin m=0.6, |N|=50)
5. LightGCN-Softmax (tau=0.15) (Sampled Cosine Softmax Contrastive Loss, tau=0.15, |N|=50)

Generates:
- metric_comparison.png: Publication-quality 5-model grouped bar chart with error bars.
- learned_weights.png: Adaptive layer weights distribution across seeds.
- comparison_summary.csv & raw_seed_metrics.csv
- comparison_results.json
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
    softmax_temperature: float = 0.15,
    num_negatives: int = 1,
    score_type: str = "dot",
    force_retrain: bool = False,
):
    """Train and evaluate a specific model variant for a given seed, or load cached results."""
    summary_path = os.path.join(save_dir, "train_summary.json")
    if not force_retrain and os.path.exists(summary_path):
        print(f"Loading cached results for {model_name} (Seed {seed}) from {summary_path}...", flush=True)
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        learned_weights = None
        best_model_path = os.path.join(save_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            ckpt = torch.load(best_model_path, map_location="cpu")
            if "model_state_dict" in ckpt and "layer_logits" in ckpt["model_state_dict"]:
                logits = ckpt["model_state_dict"]["layer_logits"]
                learned_weights = torch.softmax(logits, dim=0).numpy().tolist()
        return summary["test_metrics"], {}, learned_weights

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
        softmax_temperature=softmax_temperature,
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
):
    """Generate multi-model comparison bar chart and learned layer weight plots."""
    os.makedirs(save_dir, exist_ok=True)

    metrics = ["Recall@10", "Recall@20", "NDCG@10", "NDCG@20"]
    x = np.arange(len(metrics))
    models = results_df.index.tolist()
    num_models = len(models)

    # Color palette
    color_palette = {
        "LightGCN (BPR)": "#4C72B0",
        "AdaptiveLightGCN (BPR)": "#55A868",
        "LightGCN-CCL (m=0.8)": "#C44E52",
        "LightGCN-CCL (m=0.6)": "#8172B3",
        "LightGCN-Softmax (tau=0.15)": "#D95F02",
    }
    default_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#D95F02", "#64B5CD", "#CCB974"]

    fig, ax = plt.subplots(figsize=(12, 6.5))

    width = min(0.75 / max(num_models, 1), 0.16)

    for idx, m in enumerate(models):
        means = [results_df.loc[m, f"{metric}_mean"] for metric in metrics]
        stds = [
            results_df.loc[m, f"{metric}_std"]
            if not np.isnan(results_df.loc[m, f"{metric}_std"])
            else 0.0
            for metric in metrics
        ]
        offset = (idx - (num_models - 1) / 2.0) * width
        color = color_palette.get(m, default_colors[idx % len(default_colors)])
        rects = ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=m,
            capsize=3.5,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.92,
        )

        # Label peak values on top of bars
        for rect, mean_val in zip(rects, means):
            height = rect.get_height()
            ax.annotate(
                f"{mean_val:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
            )

    ax.set_ylabel("Evaluation Metric Score", fontsize=12, fontweight="bold")
    ax.set_title("LightGCN Architectural & Loss Variants Comparison (MovieLens-100K)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(results_df[[f"{m}_mean" for m in metrics]].max()) * 1.22)
    ax.legend(fontsize=10.5, loc="upper left", framealpha=0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plot_path = os.path.join(save_dir, "metric_comparison.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved metric comparison plot to {plot_path}", flush=True)

    # Learned Weights Visualization (for Adaptive)
    if learned_weights_list:
        weights_arr = np.array(learned_weights_list)
        mean_weights = weights_arr.mean(axis=0)
        std_weights = weights_arr.std(axis=0)
        num_layers = len(mean_weights)
        uniform_val = 1.0 / num_layers

        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        layers = [f"Layer {k}" for k in range(num_layers)]
        ax.bar(layers, mean_weights, yerr=std_weights, color="#55A868", capsize=5, edgecolor="black", linewidth=0.6, label="Learned Mean Weight")
        ax.axhline(uniform_val, color="red", linestyle="--", linewidth=1.5, label=f"Uniform Weight ({uniform_val:.3f})")

        ax.set_ylabel("Aggregation Weight (alpha_k)", fontsize=12, fontweight="bold")
        ax.set_title("AdaptiveLightGCN Learned Layer Combination Weights", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylim(0, max(mean_weights.max() * 1.35, uniform_val * 1.5))
        ax.legend(fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        weights_plot_path = os.path.join(save_dir, "learned_weights.png")
        fig.tight_layout()
        fig.savefig(weights_plot_path, dpi=300)
        plt.close(fig)
        print(f"Saved learned weights plot to {weights_plot_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run controlled comparison among LightGCN variants on MovieLens-100K."
    )
    parser.add_argument("--data_dir", type=str, default="data/ml-100k", help="Path to processed dataset.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123], help="Random seeds to evaluate.")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs per run.")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension d.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of propagation layers K.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--reg_weight", type=float, default=1e-4, help="L2 regularization.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size.")
    parser.add_argument("--save_dir", type=str, default="experiments/comparison", help="Output directory.")
    parser.add_argument("--ccl_num_negatives", type=int, default=50, help="Number of sampled negatives for contrastive variants.")
    parser.add_argument("--force_retrain", action="store_true", help="Force retrain instead of using cached runs.")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Loading data from {args.data_dir}...", flush=True)
    data = load_dataset(args.data_dir)
    norm_adj = build_torch_normalized_adjacency(data.num_users, data.num_items, data.train_pairs)

    all_results = []
    adaptive_weights = []

    model_configs = [
        # (model_name, model_cls, loss_type, ccl_margin, softmax_tau, num_negs, score_type, subdir_prefix)
        ("LightGCN (BPR)", LightGCN, "bpr", 0.8, 0.15, 1, "dot", "baseline"),
        ("AdaptiveLightGCN (BPR)", AdaptiveLightGCN, "bpr", 0.8, 0.15, 1, "dot", "adaptive"),
        ("LightGCN-CCL (m=0.8)", LightGCNCCL, "ccl", 0.8, 0.15, args.ccl_num_negatives, "cosine", "lightgcn_ccl"),
        ("LightGCN-CCL (m=0.6)", LightGCNCCL, "ccl", 0.6, 0.15, args.ccl_num_negatives, "cosine", "lightgcn_ccl_m06"),
        ("LightGCN-Softmax (tau=0.15)", LightGCNCCL, "cosine_softmax", 0.8, 0.15, args.ccl_num_negatives, "cosine", "lightgcn_softmax_tau015"),
    ]

    for seed in args.seeds:
        print(f"\n================ Running Seed {seed} ================", flush=True)

        for model_name, model_cls, loss_type, margin, tau, num_negs, score_type, subdir_prefix in model_configs:
            model_save_dir = os.path.join(args.save_dir, f"{subdir_prefix}_seed_{seed}")
            print(f"\n--- Model: {model_name} (Seed {seed}) ---", flush=True)

            test_metrics, _, weights = run_experiment_for_model(
                model_cls=model_cls,
                model_name=model_name,
                data=data,
                norm_adj=norm_adj,
                seed=seed,
                epochs=args.epochs,
                embedding_dim=args.embedding_dim,
                num_layers=args.num_layers,
                lr=args.learning_rate,
                reg_weight=args.reg_weight,
                batch_size=args.batch_size,
                save_dir=model_save_dir,
                loss_type=loss_type,
                ccl_margin=margin,
                ccl_negative_weight=10.0,
                softmax_temperature=tau,
                num_negatives=num_negs,
                score_type=score_type,
                force_retrain=args.force_retrain,
            )

            row = {"model": model_name, "seed": seed} | test_metrics
            all_results.append(row)
            if weights and model_name == "AdaptiveLightGCN (BPR)":
                adaptive_weights.append(weights)

    # Compile results
    raw_df = pd.DataFrame(all_results)
    raw_csv_path = os.path.join(args.save_dir, "raw_seed_metrics.csv")
    raw_df.to_csv(raw_csv_path, index=False)

    # Maintain model order
    order = [cfg[0] for cfg in model_configs]
    agg_df = raw_df.groupby("model")[["Recall@10", "Recall@20", "NDCG@10", "NDCG@20"]].agg(["mean", "std"])

    summary_df = pd.DataFrame(index=order)
    for col in ["Recall@10", "Recall@20", "NDCG@10", "NDCG@20"]:
        summary_df[f"{col}_mean"] = [agg_df.loc[m, (col, "mean")] if m in agg_df.index else np.nan for m in order]
        summary_df[f"{col}_std"] = [agg_df.loc[m, (col, "std")] if m in agg_df.index else np.nan for m in order]

    summary_csv_path = os.path.join(args.save_dir, "comparison_summary.csv")
    summary_df.to_csv(summary_csv_path)

    print("\n================ FINAL COMPARISON SUMMARY ================", flush=True)
    print(summary_df.round(4), flush=True)

    # Generate plots
    plot_comparison(summary_df, adaptive_weights, args.save_dir)

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
        "ccl_num_negatives": args.ccl_num_negatives,
        "models_evaluated": order,
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
