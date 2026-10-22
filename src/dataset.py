"""Dataset processing module for LightGCN.

Handles:
1. Converting raw user and item identifiers into contiguous integer indices [0, num_users - 1]
   and [0, num_items - 1].
2. Converting explicit ratings (e.g. 1-5 stars) to implicit binary feedback (interacted vs not).
3. Deduplicating user-item interactions based on documented criteria.
4. Splitting interactions into train, validation, and test sets without data leakage.
5. Computing and logging dataset statistics (users, items, interactions, density, sparsity).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import pandas as pd


@dataclass
class DatasetStats:
    """Summary statistics for an interaction dataset."""
    num_users: int
    num_items: int
    num_train_interactions: int
    num_val_interactions: int
    num_test_interactions: int
    total_interactions: int
    density: float
    sparsity: float

    def __str__(self) -> str:
        return (
            f"Dataset Statistics:\n"
            f"  - Users: {self.num_users:,}\n"
            f"  - Items: {self.num_items:,}\n"
            f"  - Train interactions: {self.num_train_interactions:,}\n"
            f"  - Validation interactions: {self.num_val_interactions:,}\n"
            f"  - Test interactions: {self.num_test_interactions:,}\n"
            f"  - Total unique interactions: {self.total_interactions:,}\n"
            f"  - Density: {self.density:.6f} ({self.density * 100:.4f}%)\n"
            f"  - Sparsity: {self.sparsity:.6f} ({self.sparsity * 100:.4f}%)"
        )


@dataclass
class InteractionData:
    """Container for processed user-item interaction data."""
    num_users: int
    num_items: int
    train_pairs: np.ndarray  # Shape: (num_train, 2), columns: [user_id, item_id]
    val_pairs: np.ndarray    # Shape: (num_val, 2), columns: [user_id, item_id]
    test_pairs: np.ndarray   # Shape: (num_test, 2), columns: [user_id, item_id]
    user_to_id: Dict[Any, int] = field(default_factory=dict)
    id_to_user: Dict[int, Any] = field(default_factory=dict)
    item_to_id: Dict[Any, int] = field(default_factory=dict)
    id_to_item: Dict[int, Any] = field(default_factory=dict)

    # Fast lookup dictionaries mapping user_id -> set of interacted item_ids
    train_user_positives: Dict[int, Set[int]] = field(default_factory=dict)
    val_user_positives: Dict[int, Set[int]] = field(default_factory=dict)
    test_user_positives: Dict[int, Set[int]] = field(default_factory=dict)

    def __post_init__(self):
        """Build user-to-item sets for train, validation, and test."""
        if not self.train_user_positives and len(self.train_pairs) > 0:
            self.train_user_positives = self._build_user_dict(self.train_pairs)
        if not self.val_user_positives and len(self.val_pairs) > 0:
            self.val_user_positives = self._build_user_dict(self.val_pairs)
        if not self.test_user_positives and len(self.test_pairs) > 0:
            self.test_user_positives = self._build_user_dict(self.test_pairs)

    @staticmethod
    def _build_user_dict(pairs: np.ndarray) -> Dict[int, Set[int]]:
        user_dict: Dict[int, Set[int]] = {}
        for u, i in pairs:
            u_int, i_int = int(u), int(i)
            if u_int not in user_dict:
                user_dict[u_int] = set()
            user_dict[u_int].add(i_int)
        return user_dict

    def get_stats(self) -> DatasetStats:
        """Calculate and return dataset statistics."""
        num_train = len(self.train_pairs)
        num_val = len(self.val_pairs)
        num_test = len(self.test_pairs)
        total = num_train + num_val + num_test
        total_possible = self.num_users * self.num_items
        density = (total / total_possible) if total_possible > 0 else 0.0
        sparsity = 1.0 - density
        return DatasetStats(
            num_users=self.num_users,
            num_items=self.num_items,
            num_train_interactions=num_train,
            num_val_interactions=num_val,
            num_test_interactions=num_test,
            total_interactions=total,
            density=density,
            sparsity=sparsity,
        )


def process_interaction_df(
    df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: Optional[str] = None,
    threshold: Optional[float] = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    min_user_interactions: int = 0,
    min_item_interactions: int = 0,
    split_strategy: str = "user_stratified",
    seed: int = 42,
) -> InteractionData:
    """Process an interaction DataFrame into an InteractionData object.

    Args:
        df: Input DataFrame containing user-item interactions.
        user_col: Name of column containing user IDs.
        item_col: Name of column containing item IDs.
        rating_col: Optional column name for explicit rating values.
        threshold: If provided, ratings >= threshold are kept as positive interactions.
                   If None and rating_col is provided, all observed ratings are treated
                   as positive feedback.
        val_ratio: Proportion of interactions reserved for validation.
        test_ratio: Proportion of interactions reserved for testing.
        min_user_interactions: Filter out users with fewer interactions (k-core).
        min_item_interactions: Filter out items with fewer interactions (k-core).
        split_strategy: "user_stratified" (splits each user's history) or "random".
        seed: Random seed for reproducible splitting.

    Returns:
        InteractionData containing mapped IDs, splits, and lookup dictionaries.
    """
    if val_ratio < 0.0 or test_ratio < 0.0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError(
            f"Invalid split ratios: val_ratio={val_ratio}, test_ratio={test_ratio}. "
            f"Sum must be strictly less than 1.0."
        )

    data = df[[user_col, item_col]].copy()
    if rating_col is not None and rating_col in df.columns:
        data[rating_col] = df[rating_col]
        if threshold is not None:
            # Filter explicit ratings to implicit positive threshold
            data = data[data[rating_col] >= threshold].copy()

    # Rule: Deduplicate (user, item) pairs. If explicit rating exists, keep highest rating.
    if rating_col is not None and rating_col in data.columns:
        data = data.sort_values(by=rating_col, ascending=False)
    data = data.drop_duplicates(subset=[user_col, item_col], keep="first")

    # Optional k-core iterative filtering
    if min_user_interactions > 0 or min_item_interactions > 0:
        prev_len = len(data)
        while True:
            if min_user_interactions > 0:
                user_counts = data[user_col].value_counts()
                valid_users = user_counts[user_counts >= min_user_interactions].index
                data = data[data[user_col].isin(valid_users)]
            if min_item_interactions > 0:
                item_counts = data[item_col].value_counts()
                valid_items = item_counts[item_counts >= min_item_interactions].index
                data = data[data[item_col].isin(valid_items)]
            if len(data) == prev_len:
                break
            prev_len = len(data)

    # Remap users and items to contiguous [0, M-1] and [0, N-1]
    unique_users = sorted(data[user_col].unique())
    unique_items = sorted(data[item_col].unique())

    user_to_id = {u: idx for idx, u in enumerate(unique_users)}
    id_to_user = {idx: u for idx, u in enumerate(unique_users)}
    item_to_id = {i: idx for idx, i in enumerate(unique_items)}
    id_to_item = {idx: i for idx, i in enumerate(unique_items)}

    data["u_idx"] = data[user_col].map(user_to_id)
    data["i_idx"] = data[item_col].map(item_to_id)

    rng = np.random.RandomState(seed)

    train_rows: List[Tuple[int, int]] = []
    val_rows: List[Tuple[int, int]] = []
    test_rows: List[Tuple[int, int]] = []

    if split_strategy == "user_stratified":
        # Split interactions per user to guarantee users in val/test have training history
        grouped = data.groupby("u_idx")["i_idx"].apply(list).to_dict()
        for u_idx, items in grouped.items():
            items = np.array(items, dtype=np.int64)
            rng.shuffle(items)
            n_items = len(items)
            n_test = int(np.floor(n_items * test_ratio))
            n_val = int(np.floor(n_items * val_ratio))

            # Ensure at least 1 item remains in training for this user if n_items >= 3
            if n_items >= 3 and (n_test + n_val) >= n_items:
                n_test = 1
                n_val = 1

            test_items = items[:n_test]
            val_items = items[n_test : n_test + n_val]
            train_items = items[n_test + n_val :]

            for it in train_items:
                train_rows.append((u_idx, int(it)))
            for it in val_items:
                val_rows.append((u_idx, int(it)))
            for it in test_items:
                test_rows.append((u_idx, int(it)))

    elif split_strategy == "random":
        indices = np.arange(len(data))
        rng.shuffle(indices)
        n_total = len(indices)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)

        test_idx = indices[:n_test]
        val_idx = indices[n_test : n_test + n_val]
        train_idx = indices[n_test + n_val :]

        test_rows = list(zip(data.iloc[test_idx]["u_idx"], data.iloc[test_idx]["i_idx"]))
        val_rows = list(zip(data.iloc[val_idx]["u_idx"], data.iloc[val_idx]["i_idx"]))
        train_rows = list(zip(data.iloc[train_idx]["u_idx"], data.iloc[train_idx]["i_idx"]))
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")

    train_pairs = np.array(train_rows, dtype=np.int64) if train_rows else np.empty((0, 2), dtype=np.int64)
    val_pairs = np.array(val_rows, dtype=np.int64) if val_rows else np.empty((0, 2), dtype=np.int64)
    test_pairs = np.array(test_rows, dtype=np.int64) if test_rows else np.empty((0, 2), dtype=np.int64)

    return InteractionData(
        num_users=len(unique_users),
        num_items=len(unique_items),
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        user_to_id=user_to_id,
        id_to_user=id_to_user,
        item_to_id=item_to_id,
        id_to_item=id_to_item,
    )


def load_benchmark_txt(
    train_file: str,
    test_file: str,
    val_file: Optional[str] = None,
    val_ratio_from_train: float = 0.0,
    seed: int = 42,
) -> InteractionData:
    """Load benchmark datasets formatted as user item1 item2 ... per line (e.g. Gowalla / Yelp).

    Format per line:
    <user_id> <item_id_1> <item_id_2> ...
    """
    def parse_file(path: str) -> List[Tuple[int, int]]:
        pairs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                user_id = int(parts[0])
                for item_id_str in parts[1:]:
                    pairs.append((user_id, int(item_id_str)))
        return pairs

    train_raw = parse_file(train_file)
    test_raw = parse_file(test_file)
    val_raw = parse_file(val_file) if val_file is not None else []

    all_pairs = train_raw + test_raw + val_raw
    all_users = set(p[0] for p in all_pairs)
    all_items = set(p[1] for p in all_pairs)

    num_users = max(all_users) + 1 if all_users else 0
    num_items = max(all_items) + 1 if all_items else 0

    train_pairs = np.array(train_raw, dtype=np.int64)
    test_pairs = np.array(test_raw, dtype=np.int64)
    val_pairs = np.array(val_raw, dtype=np.int64) if val_raw else np.empty((0, 2), dtype=np.int64)

    # If no explicit validation file provided but val_ratio_from_train > 0:
    if len(val_pairs) == 0 and val_ratio_from_train > 0.0:
        rng = np.random.RandomState(seed)
        shuffled = rng.permutation(len(train_pairs))
        n_val = int(len(train_pairs) * val_ratio_from_train)
        val_pairs = train_pairs[shuffled[:n_val]]
        train_pairs = train_pairs[shuffled[n_val:]]

    return InteractionData(
        num_users=num_users,
        num_items=num_items,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
    )
