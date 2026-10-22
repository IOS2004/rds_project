"""Negative sampling module for BPR training.

Constructs training triples (u, i, j) where:
- u is a user ID.
- i is an observed positive item from the training set.
- j is an unobserved item sampled uniformly from {0, ..., num_items - 1} \\ N_u.
"""

from typing import Dict, List, Set, Tuple, Iterator, Optional
import numpy as np
import torch


class BPRSampler:
    """Uniform negative sampler for BPR and CCL training.

    Supports sampling 1 or multiple negative items per positive interaction.
    Protocol:
    - Uniform negative sampling from unobserved items ({0, ..., num_items - 1} \\ N_u).
    - For multi-negative sampling (num_negatives > 1), sampling is performed with replacement
      across candidate draws, ensuring valid sampling even when num_negatives exceeds
      the count of remaining unobserved items for dense users.
    - Observed training positive items are strictly excluded.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        train_pairs: np.ndarray,
        train_user_positives: Dict[int, Set[int]],
        batch_size: int = 1024,
        num_negatives: int = 1,
        seed: int = 42,
        sample_per_batch: Optional[bool] = None,
    ) -> None:
        """Initialize BPR/CCL negative sampler.

        Args:
            num_users: Total number of users M.
            num_items: Total number of items N.
            train_pairs: Array of shape (num_train, 2) containing [u, i] pairs.
            train_user_positives: Mapping of user_id -> set of training positive item IDs.
            batch_size: Size of mini-batches for training.
            num_negatives: Number of negative items to sample per positive interaction (default: 1).
            seed: Random seed for reproducible negative sampling.
            sample_per_batch: If True, sample negatives dynamically per mini-batch to save memory.
                              Defaults to True when num_negatives > 1, False when num_negatives == 1.
        """
        if num_negatives < 1:
            raise ValueError(f"num_negatives must be an integer >= 1, got {num_negatives}")

        self.num_users = num_users
        self.num_items = num_items
        self.train_pairs = np.asarray(train_pairs, dtype=np.int64)
        self.train_user_positives = train_user_positives
        self.batch_size = batch_size
        self.num_negatives = int(num_negatives)
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.num_interactions = len(self.train_pairs)

        # Default sample_per_batch: on-the-fly for multi-negative to avoid O(N_train * num_negatives) memory
        if sample_per_batch is None:
            self.sample_per_batch = self.num_negatives > 1
        else:
            self.sample_per_batch = bool(sample_per_batch)

    def sample_negative_item(self, user_id: int) -> int:
        """Sample a single negative item for user_id excluding positive training items."""
        positives = self.train_user_positives.get(user_id, set())
        if len(positives) >= self.num_items:
            raise ValueError(
                f"User {user_id} has interacted with all {self.num_items} items; "
                f"no eligible negative items exist."
            )

        # Rejection sampling: draw until finding an item not in positives
        while True:
            candidate = int(self.rng.randint(0, self.num_items))
            if candidate not in positives:
                return candidate

    def sample_negative_items(self, user_id: int, num_negatives: int) -> np.ndarray:
        """Sample multiple negative items for user_id excluding positive training items.

        Sampling is uniform with replacement across eligible unobserved items.

        Args:
            user_id: ID of the user.
            num_negatives: Number of negative item IDs to sample.

        Returns:
            neg_ids: 1D numpy array of length num_negatives containing valid negative item IDs.
        """
        if num_negatives < 1:
            raise ValueError(f"num_negatives must be an integer >= 1, got {num_negatives}")

        positives = self.train_user_positives.get(user_id, set())
        num_pos = len(positives)

        if num_pos >= self.num_items:
            raise ValueError(
                f"User {user_id} has interacted with all {self.num_items} items; "
                f"no eligible negative items exist."
            )

        # Dense user case: if user interacted with > 50% of items, enumerate eligible unobserved items directly
        if num_pos > self.num_items * 0.5:
            all_items_set = set(range(self.num_items))
            eligible = np.array(list(all_items_set - positives), dtype=np.int64)
            return self.rng.choice(eligible, size=num_negatives, replace=True)

        # Sparse user case (standard in CF): vectorized rejection sampling with chunked oversampling
        collected: List[int] = []
        while len(collected) < num_negatives:
            needed = num_negatives - len(collected)
            # Oversample by 30% to account for collisions with positive interactions
            draw_size = max(int(needed * 1.3) + 5, 16)
            candidates = self.rng.randint(0, self.num_items, size=draw_size)
            valid = [int(c) for c in candidates if c not in positives]
            collected.extend(valid[:needed])

        return np.array(collected[:num_negatives], dtype=np.int64)

    def sample_epoch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a complete set of training interactions and sampled negatives for one epoch.

        Returns:
            users: Array of user IDs, shape (num_interactions,).
            pos_items: Array of positive item IDs, shape (num_interactions,).
            neg_items: Array of negative item IDs:
                       - shape (num_interactions,) if num_negatives == 1
                       - shape (num_interactions, num_negatives) if num_negatives > 1
        """
        # Shuffle training pairs
        indices = self.rng.permutation(self.num_interactions)
        shuffled_pairs = self.train_pairs[indices]

        users = shuffled_pairs[:, 0]
        pos_items = shuffled_pairs[:, 1]

        if self.num_negatives == 1:
            neg_items = np.empty(self.num_interactions, dtype=np.int64)
            for idx, u in enumerate(users):
                neg_items[idx] = self.sample_negative_item(int(u))
        else:
            neg_items = np.empty((self.num_interactions, self.num_negatives), dtype=np.int64)
            for idx, u in enumerate(users):
                neg_items[idx] = self.sample_negative_items(int(u), self.num_negatives)

        return users, pos_items, neg_items

    def iterate_batches(
        self, device: torch.device = torch.device("cpu")
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield mini-batches of (users, pos_items, neg_items) as PyTorch tensors.

        Uses on-the-fly sampling per batch if sample_per_batch is True to maintain O(B) memory.
        """
        if self.sample_per_batch:
            # Memory-efficient on-the-fly batch sampling
            indices = self.rng.permutation(self.num_interactions)
            shuffled_pairs = self.train_pairs[indices]
            num_batches = int(np.ceil(self.num_interactions / self.batch_size))

            for b in range(num_batches):
                start = b * self.batch_size
                end = min(start + self.batch_size, self.num_interactions)
                batch_pairs = shuffled_pairs[start:end]
                batch_users = batch_pairs[:, 0]
                batch_pos = batch_pairs[:, 1]

                if self.num_negatives == 1:
                    batch_neg = np.empty(len(batch_users), dtype=np.int64)
                    for idx, u in enumerate(batch_users):
                        batch_neg[idx] = self.sample_negative_item(int(u))
                else:
                    batch_neg = np.empty((len(batch_users), self.num_negatives), dtype=np.int64)
                    for idx, u in enumerate(batch_users):
                        batch_neg[idx] = self.sample_negative_items(int(u), self.num_negatives)

                b_u = torch.from_numpy(batch_users).to(device=device, dtype=torch.long)
                b_pos = torch.from_numpy(batch_pos).to(device=device, dtype=torch.long)
                b_neg = torch.from_numpy(batch_neg).to(device=device, dtype=torch.long)
                yield b_u, b_pos, b_neg
        else:
            # Pre-sampled epoch array (standard BPR single-negative path)
            users, pos_items, neg_items = self.sample_epoch()
            num_batches = int(np.ceil(self.num_interactions / self.batch_size))

            for b in range(num_batches):
                start = b * self.batch_size
                end = min(start + self.batch_size, self.num_interactions)

                b_u = torch.from_numpy(users[start:end]).to(device=device, dtype=torch.long)
                b_pos = torch.from_numpy(pos_items[start:end]).to(device=device, dtype=torch.long)
                b_neg = torch.from_numpy(neg_items[start:end]).to(device=device, dtype=torch.long)

                yield b_u, b_pos, b_neg


# Semantic alias
UniformNegativeSampler = BPRSampler
