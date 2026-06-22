"""Generate deterministic synthetic users from the held-out arXiv test set."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_PATH = PROJECT_ROOT / "arxiv_dataset" / "test.csv"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "outputs" / "synthetic_test_users.json"

USER_ARCHETYPES = {
    "CV_researcher": {
        "primary": ["cs.CV"],
        "secondary": ["cs.LG", "cs.RO"],
    },
    "NLP_researcher": {
        "primary": ["cs.CL"],
        "secondary": ["cs.IR", "cs.LG", "cs.AI"],
    },
    "ML_theorist": {
        "primary": ["cs.LG", "stat.ML"],
        "secondary": ["cs.AI", "math.OC"],
    },
    "AI_researcher": {
        "primary": ["cs.AI"],
        "secondary": ["cs.LG", "cs.CL"],
    },
    "multimodal_researcher": {
        "primary": ["cs.CV", "cs.CL"],
        "secondary": ["cs.LG", "cs.AI"],
    },
}

USER_TYPES = {
    "new_user": {"n_papers_range": (3, 8), "weight": 0.20},
    "regular": {"n_papers_range": (10, 25), "weight": 0.50},
    "power_user": {"n_papers_range": (30, 60), "weight": 0.30},
}


def build_category_index(df: pd.DataFrame) -> dict[str, list[str]]:
    """Map each primary category to paper IDs in dataset row order."""
    required = {"paper_id", "primary_category"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if not df["paper_id"].is_unique:
        raise ValueError("paper_id values must be unique.")
    if df[["paper_id", "primary_category"]].isna().any().any():
        raise ValueError("paper_id and primary_category must not contain missing values.")

    index: dict[str, list[str]] = defaultdict(list)
    for paper_id, category in df[["paper_id", "primary_category"]].itertuples(
        index=False, name=None
    ):
        index[str(category)].append(str(paper_id))
    return dict(index)


def available_archetypes(
    category_index: dict[str, list[str]],
) -> dict[str, dict[str, list[str]]]:
    """Filter the notebook archetypes to categories available in the test set."""
    result: dict[str, dict[str, list[str]]] = {}
    for name, archetype in USER_ARCHETYPES.items():
        primary = [cat for cat in archetype["primary"] if cat in category_index]
        secondary = [
            cat for cat in archetype["secondary"] if cat in category_index
        ]
        if primary:
            result[name] = {"primary": primary, "secondary": secondary}
    if not result:
        raise ValueError("No configured user archetype is available in the dataset.")
    return result


def sample_papers_for_archetype(
    all_paper_ids: Sequence[str],
    category_index: dict[str, list[str]],
    archetype: dict[str, list[str]],
    n_papers: int,
    rng: random.Random,
) -> list[str]:
    """Sample 70% primary, 20% secondary and 10% noise papers."""
    if n_papers < 2:
        raise ValueError("n_papers must be at least 2 for leave-one-out.")
    if n_papers > len(all_paper_ids):
        raise ValueError("n_papers exceeds the number of unique papers.")

    n_primary = max(1, int(n_papers * 0.70))
    n_secondary = (
        max(0, int(n_papers * 0.20)) if archetype["secondary"] else 0
    )
    n_noise = n_papers - n_primary - n_secondary

    primary_pool = [
        paper_id
        for category in archetype["primary"]
        for paper_id in category_index.get(category, [])
    ]
    secondary_pool = [
        paper_id
        for category in archetype["secondary"]
        for paper_id in category_index.get(category, [])
    ]
    if not primary_pool:
        raise ValueError("Archetype has no primary-category papers.")

    sampled = rng.choices(primary_pool, k=n_primary)
    if n_secondary and secondary_pool:
        sampled.extend(rng.choices(secondary_pool, k=n_secondary))
    else:
        n_noise += n_secondary
    sampled.extend(rng.choices(list(all_paper_ids), k=n_noise))

    unique_papers = list(dict.fromkeys(sampled))
    seen = set(unique_papers)
    remaining = [paper_id for paper_id in all_paper_ids if paper_id not in seen]
    rng.shuffle(remaining)
    unique_papers.extend(remaining[: n_papers - len(unique_papers)])
    return unique_papers


def generate_days_ago(n_papers: int, np_rng: np.random.Generator) -> list[int]:
    """Generate the notebook's exponential reading-time gaps in [0, 180]."""
    if n_papers <= 0:
        raise ValueError("n_papers must be positive.")
    if n_papers == 1:
        return [0]

    gaps = np_rng.exponential(scale=45 / (n_papers - 1), size=n_papers - 1)
    gaps = np.clip(gaps, 0.5, None)
    cumulative = np.cumsum(gaps[::-1])[::-1]
    return np.clip(np.append(cumulative, 0), 0, 180).astype(int).tolist()


def get_negative_papers(
    df: pd.DataFrame,
    target_paper: str,
    primary_categories: Sequence[str],
    n_negatives: int,
    rng: random.Random,
) -> list[str]:
    """Sample negatives outside the archetype's primary categories."""
    if n_negatives < 0:
        raise ValueError("n_negatives must be non-negative.")
    primary = set(primary_categories)
    negative_pool = df.loc[
        (df["paper_id"] != target_paper)
        & (~df["primary_category"].isin(primary)),
        "paper_id",
    ].astype(str).tolist()
    if len(negative_pool) < n_negatives:
        negative_pool = df.loc[
            df["paper_id"] != target_paper, "paper_id"
        ].astype(str).tolist()
    return rng.sample(negative_pool, min(n_negatives, len(negative_pool)))


def generate_synthetic_test_users(
    df: pd.DataFrame,
    n_users: int = 500,
    n_negatives: int = 6,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate test users with the same design as the project notebook."""
    if n_users <= 0:
        raise ValueError("n_users must be positive.")

    category_index = build_category_index(df)
    archetypes = available_archetypes(category_index)
    all_paper_ids = df["paper_id"].astype(str).tolist()
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    user_type_names = list(USER_TYPES)
    user_type_weights = [USER_TYPES[name]["weight"] for name in user_type_names]
    archetype_names = list(archetypes)

    users: list[dict[str, Any]] = []
    for user_index in range(n_users):
        archetype_name = rng.choice(archetype_names)
        archetype = archetypes[archetype_name]
        user_type = rng.choices(
            user_type_names, weights=user_type_weights, k=1
        )[0]
        lower, upper = USER_TYPES[user_type]["n_papers_range"]
        n_papers = rng.randint(lower, min(upper, len(all_paper_ids)))

        paper_ids = sample_papers_for_archetype(
            all_paper_ids,
            category_index,
            archetype,
            n_papers,
            rng,
        )
        days_ago = generate_days_ago(len(paper_ids), np_rng)
        reading_history = sorted(
            zip(paper_ids, days_ago), key=lambda item: item[1], reverse=True
        )
        train_history = [
            [paper_id, int(days)] for paper_id, days in reading_history[:-1]
        ]
        target_paper = reading_history[-1][0]
        negatives = get_negative_papers(
            df,
            target_paper,
            archetype["primary"],
            n_negatives,
            rng,
        )
        users.append(
            {
                "user_id": f"test_user_{user_index:04d}",
                "archetype": archetype_name,
                "user_type": user_type,
                "train_history": train_history,
                "target_paper": target_paper,
                "negative_papers": negatives,
            }
        )
    return users


def validate_users(users: Sequence[dict[str, Any]], df: pd.DataFrame) -> None:
    """Reject leakage, invalid times and papers outside the test corpus."""
    dataset_ids = set(df["paper_id"].astype(str))
    for user in users:
        history = user["train_history"]
        history_ids = [paper_id for paper_id, _ in history]
        days = [days_ago for _, days_ago in history]
        target = user["target_paper"]
        negatives = user["negative_papers"]

        if not history:
            raise ValueError(f"{user['user_id']} has an empty history.")
        if len(history_ids) != len(set(history_ids)):
            raise ValueError(f"{user['user_id']} contains duplicate history papers.")
        if target in history_ids or target in negatives:
            raise ValueError(f"{user['user_id']} contains target leakage.")
        if not set(history_ids + [target] + negatives).issubset(dataset_ids):
            raise ValueError(f"{user['user_id']} contains a paper outside test.csv.")
        if days != sorted(days, reverse=True):
            raise ValueError(f"{user['user_id']} history is not oldest-first.")
        if any(day < 0 or day > 180 for day in days):
            raise ValueError(f"{user['user_id']} has invalid days_ago.")


def save_users(
    users: Sequence[dict[str, Any]],
    output_path: Path,
    seed: int,
    n_negatives: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "n_users": len(users),
            "n_negatives": n_negatives,
            "random_seed": seed,
            "source": "arxiv_dataset/test.csv",
            "protocol": "archetype sampling with leave-one-out",
        },
        "test": list(users),
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic leave-one-out users from test.csv."
    )
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--n-users", type=int, default=500)
    parser.add_argument("--n-negatives", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.test_data)
    users = generate_synthetic_test_users(
        df,
        n_users=args.n_users,
        n_negatives=args.n_negatives,
        seed=args.seed,
    )
    validate_users(users, df)
    save_users(users, args.output, args.seed, args.n_negatives)
    print(f"Generated {len(users)} users: {args.output.resolve()}")


if __name__ == "__main__":
    main()
