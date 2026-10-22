"""Evaluation metrics module for top-N recommendation.

Implements:
1. Recall@K:
       Recall@K(u) = |R_u(K) \\cap T_u| / |T_u|
2. NDCG@K:
       DCG@K(u)  = \\sum_{p=1}^K (r_p / \\log_2(p + 1))
       IDCG@K(u) = \\sum_{p=1}^{\\min(|T_u|, K)} (1 / \\log_2(p + 1))
       NDCG@K(u) = DCG@K(u) / IDCG@K(u)

Protocol:
- All unobserved items are ranking candidates.
- Training items are strictly excluded from recommendation candidates (masked to -inf).
- Metrics are averaged across evaluated users who have at least one ground-truth item.
"""

from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_recall_at_k(recommended_items: List[int], ground_truth_items: Set[int], k: int) -> float:
    """Compute Recall@K for a single user."""
    if not ground_truth_items:
        return 0.0
    top_k = recommended_items[:k]
    hits = sum(1 for item in top_k if item in ground_truth_items)
    return hits / float(len(ground_truth_items))


def compute_ndcg_at_k(recommended_items: List[int], ground_truth_items: Set[int], k: int) -> float:
    """Compute NDCG@K for a single user."""
    if not ground_truth_items:
        return 0.0

    top_k = recommended_items[:k]
    dcg = 0.0
    for p, item in enumerate(top_k):
        if item in ground_truth_items:
            # Rank p is 0-indexed, so formula uses log2((p+1) + 1) = log2(p + 2)
            dcg += 1.0 / np.log2(p + 2)

    # Ideal DCG: top items all match up to min(|T_u|, k)
    max_hits = min(len(ground_truth_items), k)
    idcg = sum(1.0 / np.log2(p + 2) for p in range(max_hits))

    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    norm_adj: torch.Tensor,
    eval_user_positives: Dict[int, Set[int]],
    train_user_positives: Dict[int, Set[int]],
    k_list: Tuple[int, ...] = (10, 20),
    user_batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
    score_type: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate a recommendation model on test or validation users across top-N metrics.

    Args:
        model: Recommendation model implementing forward_propagation or predict_all_for_users.
        norm_adj: Symmetrically normalized adjacency matrix \\tilde{A}.
        eval_user_positives: Mapping of user_id -> ground-truth items to evaluate.
        train_user_positives: Mapping of user_id -> training items to exclude from candidates.
        k_list: Tuple of K cutoffs (e.g. 10, 20).
        user_batch_size: Batch size of users evaluated concurrently.
        device: Device on which evaluation takes place.
        score_type: Scoring method: 'cosine' (L2 normalized dot product) or 'dot' (unnormalized).
                    If None, inferred from model.score_type or defaults to 'dot'.

    Returns:
        metrics: Dictionary of metric names mapped to mean scores across evaluated users.
                 E.g. {"Recall@10": ..., "Recall@20": ..., "NDCG@10": ..., "NDCG@20": ...}
    """
    model.eval()

    if score_type is None:
        score_type = getattr(model, "score_type", "dot")

    # Users who have at least one ground-truth interaction in evaluation set
    eval_users = [u for u, items in eval_user_positives.items() if len(items) > 0]
    if not eval_users:
        return {f"Recall@{k}": 0.0 for k in k_list} | {f"NDCG@{k}": 0.0 for k in k_list}

    max_k = max(k_list)

    # Precompute final user and item embeddings once to avoid repeated graph convolutions
    final_user_emb, final_item_emb, _ = model.forward_propagation(norm_adj)
    if score_type == "cosine":
        final_user_emb = F.normalize(final_user_emb, p=2, dim=-1, eps=1e-8)
        final_item_emb = F.normalize(final_item_emb, p=2, dim=-1, eps=1e-8)
    final_item_emb_t = final_item_emb.t()  # Shape: (d, N)

    accumulators: Dict[str, float] = {f"Recall@{k}": 0.0 for k in k_list}
    for k in k_list:
        accumulators[f"NDCG@{k}"] = 0.0

    num_eval_users = len(eval_users)

    for start_idx in range(0, num_eval_users, user_batch_size):
        batch_users = eval_users[start_idx : start_idx + user_batch_size]
        batch_users_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)

        # Compute dot product scores: (batch_size, d) @ (d, num_items) -> (batch_size, num_items)
        batch_u_emb = final_user_emb[batch_users_tensor]
        batch_scores = torch.matmul(batch_u_emb, final_item_emb_t)  # Shape: (B, N)

        # Exclude known training interactions by setting their predicted scores to -inf
        for row_idx, u in enumerate(batch_users):
            train_items = train_user_positives.get(u, set())
            if train_items:
                item_indices = torch.tensor(list(train_items), dtype=torch.long, device=device)
                batch_scores[row_idx, item_indices] = -float("inf")

        # Get top-max_k recommendations for each user
        _, topk_indices = torch.topk(batch_scores, k=max_k, dim=1)
        topk_indices = topk_indices.cpu().numpy()

        # Score top-K recommendations per user
        for row_idx, u in enumerate(batch_users):
            rec_items = topk_indices[row_idx].tolist()
            gt_items = eval_user_positives[u]

            for k in k_list:
                accumulators[f"Recall@{k}"] += compute_recall_at_k(rec_items, gt_items, k)
                accumulators[f"NDCG@{k}"] += compute_ndcg_at_k(rec_items, gt_items, k)

    # Compute mean across evaluated users
    result_metrics = {name: val / num_eval_users for name, val in accumulators.items()}
    return result_metrics
