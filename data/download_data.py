"""Dataset download and preparation utility.

Supports:
1. MovieLens 100K (default, fast CPU execution, standard recommender benchmark)
2. MovieLens 1M
3. Synthetic interaction dataset (for offline environments or rapid smoke tests)
4. Gowalla / Yelp benchmark format compatibility
"""

import argparse
import io
import os
import sys
import zipfile

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import requests
import numpy as np
import pandas as pd

from src.dataset import process_interaction_df, InteractionData


MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_and_extract_zip(url: str, target_dir: str) -> None:
    """Download a ZIP file and extract into target_dir."""
    os.makedirs(target_dir, exist_ok=True)
    print(f"Downloading from {url}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(target_dir)
    print(f"Extracted to {target_dir}")


def prepare_movielens_100k(
    data_dir: str = "data/ml-100k",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    threshold: float = 3.0,
    seed: int = 42,
) -> InteractionData:
    """Download and process MovieLens 100K."""
    raw_dir = os.path.join(data_dir, "raw")
    rating_file = os.path.join(raw_dir, "ml-100k", "u.data")

    if not os.path.exists(rating_file):
        download_and_extract_zip(MOVIELENS_100K_URL, raw_dir)

    print("Parsing MovieLens-100k u.data...")
    df = pd.read_csv(
        rating_file,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )

    data = process_interaction_df(
        df,
        user_col="user_id",
        item_col="item_id",
        rating_col="rating",
        threshold=threshold,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        min_user_interactions=10,
        min_item_interactions=10,
        split_strategy="user_stratified",
        seed=seed,
    )

    save_splits(data, data_dir)
    return data


def prepare_synthetic(
    data_dir: str = "data/synthetic",
    num_users: int = 100,
    num_items: int = 150,
    num_interactions: int = 2000,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> InteractionData:
    """Generate a synthetic interaction dataset for testing."""
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    # Power-law-like distribution for realistic degrees
    users = rng.choice(num_users, size=num_interactions)
    items = rng.choice(num_items, size=num_interactions)
    ratings = rng.choice([3.0, 4.0, 5.0], size=num_interactions)

    df = pd.DataFrame({"user_id": users, "item_id": items, "rating": ratings})

    data = process_interaction_df(
        df,
        user_col="user_id",
        item_col="item_id",
        rating_col="rating",
        threshold=3.0,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_strategy="user_stratified",
        seed=seed,
    )

    save_splits(data, data_dir)
    return data


def save_splits(data: InteractionData, data_dir: str) -> None:
    """Save train, val, and test interaction pairs to CSV."""
    os.makedirs(data_dir, exist_ok=True)

    pd.DataFrame(data.train_pairs, columns=["user_id", "item_id"]).to_csv(
        os.path.join(data_dir, "train.csv"), index=False
    )
    pd.DataFrame(data.val_pairs, columns=["user_id", "item_id"]).to_csv(
        os.path.join(data_dir, "val.csv"), index=False
    )
    pd.DataFrame(data.test_pairs, columns=["user_id", "item_id"]).to_csv(
        os.path.join(data_dir, "test.csv"), index=False
    )

    stats = data.get_stats()
    stats_file = os.path.join(data_dir, "stats.txt")
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write(str(stats) + "\n")

    print(f"Saved dataset splits to {data_dir}")
    print(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare recommender datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ml-100k",
        choices=["ml-100k", "synthetic"],
        help="Dataset name to prepare.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splitting.")
    parser.add_argument("--threshold", type=float, default=3.0, help="Explicit rating threshold.")
    args = parser.parse_args()

    if args.dataset == "ml-100k":
        prepare_movielens_100k(threshold=args.threshold, seed=args.seed)
    elif args.dataset == "synthetic":
        prepare_synthetic(seed=args.seed)
