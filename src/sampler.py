"""Negative sampling module for BPR training.

Constructs training triples (u, i, j) where:
- u is a user ID.
- i is an observed positive item from the training set.
- j is an unobserved item sampled uniformly from {0, ..., num_items - 1} \\ N_u.
"""

from typing import Dict, List, Set, Tuple, Iterator
import numpy as np
import torch


class BPRSampler:
    """Uniform negative sampler for BPR training."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        train_pairs: np.ndarray,
        train_user_positives: Dict[int, Set[int]],
        batch_size: int = 1024,
        seed: int = 42,
    ) -> None:
        """Initialize BPR sampler.

        Args:
            num_users: Total number of users M.
            num_items: Total number of items N.
            train_pairs: Array of shape (num_train, 2) containing [u, i] pairs.
            train_user_positives: Mapping of user_id -> set of training positive item IDs.
            batch_size: Size of mini-batches for training.
            seed: Random seed for reproducible negative sampling.
        """
        self.num_users = num_users
        self.num_items = num_items
        self.train_pairs = np.asarray(train_pairs, dtype=np.int64)
        self.train_user_positives = train_user_positives
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        self.num_interactions = len(self.train_pairs)

    def sample_negative_item(self, user_id: int) -> int:
        """Sample a single negative item for user_id excluding positive training items."""
        positives = self.train_user_positives.get(user_id, set())
        # Rejection sampling: draw until finding an item not in positives
        while True:
            candidate = int(self.rng.randint(0, self.num_items))
            if candidate not in positives:
                return candidate

    def sample_epoch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a complete set of (u, i, j) triples for one training epoch.

        Shuffles training pairs and samples one negative item j for each positive pair (u, i).

        Returns:
            users: Array of user IDs.
            pos_items: Array of positive item IDs.
            neg_items: Array of sampled negative item IDs.
        """
        # Shuffle training pairs
        indices = self.rng.permutation(self.num_interactions)
        shuffled_pairs = self.train_pairs[indices]

        users = shuffled_pairs[:, 0]
        pos_items = shuffled_pairs[:, 1]
        neg_items = np.empty(self.num_interactions, dtype=np.int64)

        for idx, u in enumerate(users):
            neg_items[idx] = self.sample_negative_item(int(u))

        return users, pos_items, neg_items

    def iterate_batches(
        self, device: torch.device = torch.device("cpu")
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield mini-batches of (users, pos_items, neg_items) as PyTorch tensors."""
        users, pos_items, neg_items = self.sample_epoch()
        num_batches = int(np.ceil(self.num_interactions / self.batch_size))

        for b in range(num_batches):
            start = b * self.batch_size
            end = min(start + self.batch_size, self.num_interactions)

            batch_u = torch.from_numpy(users[start:end]).to(device=device, dtype=torch.long)
            batch_pos = torch.from_numpy(pos_items[start:end]).to(device=device, dtype=torch.long)
            batch_neg = torch.from_numpy(neg_items[start:end]).to(device=device, dtype=torch.long)

            yield batch_u, batch_pos, batch_neg
