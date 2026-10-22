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


def test_multi_negative_sampling_shapes_and_exclusion():
    """Verify that multi-negative sampling returns expected shape, valid range, and excludes positives."""
    num_users = 3
    num_items = 20
    num_negatives = 5

    train_pairs = np.array([
        [0, 1], [0, 2], [0, 3],
        [1, 4], [1, 5],
        [2, 10], [2, 11], [2, 12],
    ], dtype=np.int64)

    user_positives = {
        0: {1, 2, 3},
        1: {4, 5},
        2: {10, 11, 12},
    }

    sampler = BPRSampler(
        num_users=num_users,
        num_items=num_items,
        train_pairs=train_pairs,
        train_user_positives=user_positives,
        batch_size=4,
        num_negatives=num_negatives,
        seed=42,
    )

    for _ in range(20):
        users, pos_items, neg_items = sampler.sample_epoch()
        assert users.shape == (len(train_pairs),)
        assert pos_items.shape == (len(train_pairs),)
        assert neg_items.shape == (len(train_pairs), num_negatives)

        for u, pos, negs in zip(users, pos_items, neg_items):
            assert pos in user_positives[u]
            assert len(negs) == num_negatives
            for neg in negs:
                assert 0 <= neg < num_items
                assert neg not in user_positives[u], f"User {u} sampled positive {neg} as negative!"


def test_multi_negative_invalid_counts():
    """Verify that requesting an invalid number of negative samples raises ValueError."""
    train_pairs = np.array([[0, 1]], dtype=np.int64)
    user_positives = {0: {1}}

    with pytest.raises(ValueError, match="num_negatives must be an integer >= 1"):
        BPRSampler(num_users=1, num_items=5, train_pairs=train_pairs, train_user_positives=user_positives, num_negatives=0)

    with pytest.raises(ValueError, match="num_negatives must be an integer >= 1"):
        BPRSampler(num_users=1, num_items=5, train_pairs=train_pairs, train_user_positives=user_positives, num_negatives=-10)


def test_multi_negative_user_interacted_all_items():
    """Verify that a user who has interacted with all catalog items raises ValueError when sampled."""
    num_items = 4
    train_pairs = np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=np.int64)
    user_positives = {0: {0, 1, 2, 3}}  # Interacted with ALL items

    sampler = BPRSampler(
        num_users=1,
        num_items=num_items,
        train_pairs=train_pairs,
        train_user_positives=user_positives,
        num_negatives=2,
    )

    with pytest.raises(ValueError, match="interacted with all"):
        sampler.sample_negative_items(user_id=0, num_negatives=2)


def test_multi_negative_batch_iteration_shapes():
    """Verify batch iteration yields correct tensor dimensions with multi-negative sampling."""
    num_users = 2
    num_items = 8
    num_negatives = 4
    batch_size = 3

    train_pairs = np.array([[0, 1], [0, 2], [1, 3], [1, 4], [0, 5]], dtype=np.int64)
    user_positives = {0: {1, 2, 5}, 1: {3, 4}}

    sampler = BPRSampler(
        num_users=num_users,
        num_items=num_items,
        train_pairs=train_pairs,
        train_user_positives=user_positives,
        batch_size=batch_size,
        num_negatives=num_negatives,
        seed=123,
    )

    batches = list(sampler.iterate_batches())
    assert len(batches) == 2  # 5 samples with batch size 3 -> 3, 2

    b1_u, b1_pos, b1_neg = batches[0]
    assert b1_u.shape == (3,)
    assert b1_pos.shape == (3,)
    assert b1_neg.shape == (3, num_negatives)

    b2_u, b2_pos, b2_neg = batches[1]
    assert b2_u.shape == (2,)
    assert b2_pos.shape == (2,)
    assert b2_neg.shape == (2, num_negatives)

