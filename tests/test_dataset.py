"""Unit tests for dataset processing and splitting."""

import numpy as np
import pandas as pd
import pytest

from src.dataset import process_interaction_df, DatasetStats


def test_id_mapping_and_deduplication():
    # User strings and item strings out of order with duplicates
    df = pd.DataFrame({
        "user_id": ["u_alice", "u_bob", "u_alice", "u_alice"],
        "item_id": ["i_book", "i_pen", "i_book", "i_apple"],
        "rating": [4.0, 5.0, 3.0, 2.0],
    })

    # Deduplication keeping highest rating for (u_alice, i_book) -> 4.0
    data = process_interaction_df(
        df,
        user_col="user_id",
        item_col="item_id",
        rating_col="rating",
        threshold=3.0,  # Excludes (u_alice, i_apple) which is 2.0
        val_ratio=0.0,
        test_ratio=0.0,
        split_strategy="random",
    )

    # Valid items: i_book (rating 4.0), i_pen (rating 5.0)
    assert data.num_users == 2  # alice, bob
    assert data.num_items == 2  # book, pen
    assert len(data.train_pairs) == 2  # (alice, book) and (bob, pen)
    assert len(data.val_pairs) == 0
    assert len(data.test_pairs) == 0

    stats = data.get_stats()
    assert stats.total_interactions == 2
    assert stats.density == 2 / (2 * 2)  # 0.5
    assert stats.sparsity == 0.5


def test_no_data_leakage_between_splits():
    # Synthetic dataset with 5 users, each having 10 items
    rows = []
    for u in range(5):
        for i in range(10):
            rows.append({"u": f"user_{u}", "i": f"item_{i}"})
    df = pd.DataFrame(rows)

    data = process_interaction_df(
        df,
        user_col="u",
        item_col="i",
        val_ratio=0.1,
        test_ratio=0.2,
        split_strategy="user_stratified",
        seed=123,
    )

    train_set = set(map(tuple, data.train_pairs))
    val_set = set(map(tuple, data.val_pairs))
    test_set = set(map(tuple, data.test_pairs))

    # Assert no overlap whatsoever between train, val, test
    assert len(train_set.intersection(val_set)) == 0, "Train and Validation sets overlap!"
    assert len(train_set.intersection(test_set)) == 0, "Train and Test sets overlap!"
    assert len(val_set.intersection(test_set)) == 0, "Validation and Test sets overlap!"

    # Total interactions equal to original unique interactions
    assert len(train_set) + len(val_set) + len(test_set) == 50


def test_reproducible_splitting():
    df = pd.DataFrame({
        "user_id": [f"u{u}" for u in range(10) for _ in range(5)],
        "item_id": [f"i{i % 20}" for i in range(50)],
    })

    data1 = process_interaction_df(df, seed=42)
    data2 = process_interaction_df(df, seed=42)
    data3 = process_interaction_df(df, seed=99)

    np.testing.assert_array_equal(data1.train_pairs, data2.train_pairs)
    np.testing.assert_array_equal(data1.test_pairs, data2.test_pairs)
    assert not np.array_equal(data1.train_pairs, data3.train_pairs)
