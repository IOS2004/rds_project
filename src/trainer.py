"""Trainer module for LightGCN models.

Manages:
1. Mini-batch BPR optimization with Adam optimizer.
2. Embedding regularization tracking.
3. Periodic validation-based model selection.
4. Early stopping on validation metric (e.g. Recall@20).
5. Saving and loading model checkpoints with hyperparameter configuration.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import InteractionData
from src.sampler import BPRSampler
from src.loss import BPRLoss
from src.metrics import evaluate_model


@dataclass
class TrainConfig:
    """Configuration dataclass for LightGCN training."""
    embedding_dim: int = 64
    num_layers: int = 3
    learning_rate: float = 0.001
    reg_weight: float = 1e-4
    batch_size: int = 1024
    epochs: int = 100
    eval_every: int = 5
    early_stop_patience: int = 10
    early_stop_metric: str = "Recall@20"
    seed: int = 42
    device: str = "cpu"


class Trainer:
    """Trainer orchestrating LightGCN optimization and evaluation."""

    def __init__(
        self,
        model: nn.Module,
        data: InteractionData,
        norm_adj: torch.Tensor,
        config: TrainConfig,
        save_dir: str = "experiments/checkpoints",
    ) -> None:
        self.model = model
        self.data = data
        self.norm_adj = norm_adj
        self.config = config
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.norm_adj = self.norm_adj.to(self.device)

        self.loss_fn = BPRLoss(reg_weight=config.reg_weight)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        self.sampler = BPRSampler(
            num_users=data.num_users,
            num_items=data.num_items,
            train_pairs=data.train_pairs,
            train_user_positives=data.train_user_positives,
            batch_size=config.batch_size,
            seed=config.seed,
        )

        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "bpr_loss": [],
            "reg_loss": [],
            "val_Recall@10": [],
            "val_Recall@20": [],
            "val_NDCG@10": [],
            "val_NDCG@20": [],
        }

    def train_epoch(self) -> Dict[str, float]:
        """Execute one complete training epoch over all mini-batches."""
        self.model.train()
        total_loss_accum = 0.0
        bpr_loss_accum = 0.0
        reg_loss_accum = 0.0
        num_batches = 0

        for batch_u, batch_pos, batch_neg in self.sampler.iterate_batches(device=self.device):
            self.optimizer.zero_grad()

            pos_scores, neg_scores, u0, pos0, neg0 = self.model(
                batch_u, batch_pos, batch_neg, self.norm_adj
            )

            loss, bpr, reg = self.loss_fn(pos_scores, neg_scores, u0, pos0, neg0)
            loss.backward()
            self.optimizer.step()

            total_loss_accum += loss.item()
            bpr_loss_accum += bpr.item()
            reg_loss_accum += reg.item()
            num_batches += 1

        return {
            "train_loss": total_loss_accum / max(num_batches, 1),
            "bpr_loss": bpr_loss_accum / max(num_batches, 1),
            "reg_loss": reg_loss_accum / max(num_batches, 1),
        }

    def evaluate(self, eval_type: str = "val") -> Dict[str, float]:
        """Evaluate current model on validation or test set."""
        eval_dict = self.data.val_user_positives if eval_type == "val" else self.data.test_user_positives
        metrics = evaluate_model(
            model=self.model,
            norm_adj=self.norm_adj,
            eval_user_positives=eval_dict,
            train_user_positives=self.data.train_user_positives,
            k_list=(10, 20),
            user_batch_size=256,
            device=self.device,
        )
        return metrics

    def train(self) -> Dict[str, Any]:
        """Run the full training loop with validation and early stopping."""
        best_metric_val = -1.0
        best_epoch = 0
        patience_counter = 0
        best_model_path = os.path.join(self.save_dir, "best_model.pt")

        print(f"Starting training for {self.config.epochs} epochs on device: {self.device}", flush=True)

        for epoch in range(1, self.config.epochs + 1):
            epoch_loss = self.train_epoch()

            # Record training metrics
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(epoch_loss["train_loss"])
            self.history["bpr_loss"].append(epoch_loss["bpr_loss"])
            self.history["reg_loss"].append(epoch_loss["reg_loss"])

            # Periodic validation evaluation
            if epoch % self.config.eval_every == 0 or epoch == self.config.epochs:
                val_metrics = self.evaluate(eval_type="val")
                for k, v in val_metrics.items():
                    self.history[f"val_{k}"].append(v)

                target_metric = val_metrics.get(self.config.early_stop_metric, 0.0)
                print(
                    f"Epoch {epoch:03d} | "
                    f"Loss: {epoch_loss['train_loss']:.4f} (BPR: {epoch_loss['bpr_loss']:.4f}, Reg: {epoch_loss['reg_loss']:.4f}) | "
                    f"Val Recall@20: {val_metrics['Recall@20']:.4f} | "
                    f"Val NDCG@20: {val_metrics['NDCG@20']:.4f}",
                    flush=True,
                )

                # Check for improvement
                if target_metric > best_metric_val:
                    best_metric_val = target_metric
                    best_epoch = epoch
                    patience_counter = 0
                    self.save_checkpoint(best_model_path, epoch=epoch, best_val=best_metric_val)
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stop_patience:
                        print(
                            f"Early stopping triggered at epoch {epoch}. "
                            f"Best {self.config.early_stop_metric}: {best_metric_val:.4f} at epoch {best_epoch}.",
                            flush=True,
                        )
                        break
            else:
                for k in ["Recall@10", "Recall@20", "NDCG@10", "NDCG@20"]:
                    self.history[f"val_{k}"].append(float("nan"))

        # Load best model for final evaluation
        if os.path.exists(best_model_path):
            self.load_checkpoint(best_model_path)

        test_metrics = self.evaluate(eval_type="test")
        print("\nFinal Test Evaluation (Best Checkpoint):")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")

        # Save training summary
        summary = {
            "config": asdict(self.config),
            "best_epoch": best_epoch,
            f"best_val_{self.config.early_stop_metric}": best_metric_val,
            "test_metrics": test_metrics,
        }
        with open(os.path.join(self.save_dir, "train_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def save_checkpoint(self, path: str, epoch: int, best_val: float) -> None:
        """Save model checkpoint with config and optimizer state."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "best_val": best_val,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model state from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
