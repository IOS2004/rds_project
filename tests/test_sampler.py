"""Unit tests for negative sampling and exclusion rules."""

import numpy as np
import pytest

from src.sampler import BPRSampler


def test_negative_sampling_exclusion_rule():
    """Verify that sampled negative items strictly exclude user's positive interactions."""
    num_users = 3
    num_items = 10

    # User 0 has interacted with items 0, 1, 2, 3
    # User 1 has interacted with items 4, 5, 6
    # User 2 has interacted with items 7, 8, 9
    train_pairs = np.array([
        [0, 0], [0, 1], [0, 2], [0, 3],
        [1, 4], [1, 5], [1, 6],
        [2, 7], [2, 8], [2, 9],
    ], dtype=np.int64)

    user_positives = {
        0: {0, 1, 2, 3},
        1: {4, 5, 6},
        2: {7, 8, 9},
    }

    sampler = BPRSampler(
        num_users=num_users,
        num_items=num_items,
        train_pairs=train_pairs,
        train_user_positives=user_positives,
        batch_size=4,
        seed=101,
    )

    # Sample repeatedly (100 epochs) to verify stochastic coverage and strict exclusion
    for _ in range(100):
        users, pos_items, neg_items = sampler.sample_epoch()
        assert len(users) == len(train_pairs)
        assert len(pos_items) == len(train_pairs)
        assert len(neg_items) == len(train_pairs)

        for u, pos, neg in zip(users, pos_items, neg_items):
            # Assert negative item is never in user's known positive items
            assert neg not in user_positives[u], f"User {u} sampled positive item {neg} as negative!"
            # Assert positive item is in user's positives
            assert pos in user_positives[u]
            # Assert negative is a valid item ID
            assert 0 <= neg < num_items


def test_sampler_batch_iteration():
    """Verify mini-batch sizes and complete iteration."""
    num_users = 2
    num_items = 5
    train_pairs = np.array([[0, 1], [0, 2], [1, 3], [1, 4], [0, 4]], dtype=np.int64)
    user_positives = {0: {1, 2, 4}, 1: {3, 4}}

    sampler = BPRSampler(num_users, num_items, train_pairs, user_positives, batch_size=2, seed=42)

    total_samples = 0
    batch_count = 0
    for batch_u, batch_pos, batch_neg in sampler.iterate_batches():
        assert len(batch_u) == len(batch_pos) == len(batch_neg)
        assert len(batch_u) <= 2
        total_samples += len(batch_u)
        batch_count += 1

    assert total_samples == 5
    assert batch_count == 3  # Batches of size 2, 2, 1
