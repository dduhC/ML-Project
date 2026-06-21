from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import scipy.sparse


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_HALF_LIVES = (30, 60, 90)


def build_paper_id_to_index(paper_ids: Sequence[str]) -> dict[str, int]:
    """Map paper IDs to the row order used by the embedding matrix."""
    if len(set(paper_ids)) != len(paper_ids):
        raise ValueError("paper_id values must be unique.")
    return {paper_id: index for index, paper_id in enumerate(paper_ids)}


def compute_temporal_weights(
    days_ago: Sequence[int | float],
    half_life: int | float,
) -> np.ndarray:
    if half_life <= 0:
        raise ValueError("half_life must be greater than zero.")
    values = np.asarray(days_ago, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("days_ago must contain at least one value.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("days_ago values must be finite and non-negative.")
    weights = np.exp(-np.log(2) * values / half_life)
    return weights / weights.sum()


def compute_user_profile(
    user: dict[str, Any],
    embeddings: np.ndarray | scipy.sparse.spmatrix,
    paper_id_to_index: dict[str, int],
    half_life: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    indices: list[int] = []
    days_ago: list[float] = []
    missing_ids: list[str] = []

    for paper_id, days in user.get("train_history", []):
        index = paper_id_to_index.get(paper_id)
        if index is None:
            missing_ids.append(paper_id)
            continue
        indices.append(index)
        days_ago.append(days)

    if not indices:
        raise ValueError(f"User {user.get('user_id')} has no valid history papers.")

    weights = compute_temporal_weights(days_ago, half_life)
    selected = embeddings[indices]
    if scipy.sparse.issparse(selected):
        profile = np.asarray(
            selected.multiply(weights[:, np.newaxis]).sum(axis=0)
        ).ravel()
    else:
        profile = np.average(np.asarray(selected), axis=0, weights=weights)

    metadata = {
        "n_papers": len(indices),
        "half_life": half_life,
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
        "missing_papers": missing_ids,
        "user_id": user["user_id"],
    }
    return np.asarray(profile, dtype=np.float32), metadata


def generate_profiles(
    users: Sequence[dict[str, Any]],
    embeddings: np.ndarray | scipy.sparse.spmatrix,
    paper_ids: Sequence[str],
    half_lives: Sequence[int] = DEFAULT_HALF_LIVES,
) -> dict[int, tuple[np.ndarray, list[dict[str, Any]]]]:
    if embeddings.ndim != 2 or embeddings.shape[0] != len(paper_ids):
        raise ValueError("Embedding rows must match paper_id row order.")

    paper_id_to_index = build_paper_id_to_index(paper_ids)
    results: dict[int, tuple[np.ndarray, list[dict[str, Any]]]] = {}
    for half_life in half_lives:
        vectors: list[np.ndarray] = []
        metadata: list[dict[str, Any]] = []
        for user in users:
            vector, user_metadata = compute_user_profile(
                user,
                embeddings,
                paper_id_to_index,
                half_life,
            )
            vectors.append(vector)
            metadata.append(user_metadata)
        results[half_life] = (np.vstack(vectors), metadata)
    return results


def save_profiles(
    output_dir: Path,
    results: dict[int, tuple[np.ndarray, list[dict[str, Any]]]],
    matrix_name: str,
    matrix_shape: tuple[int, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files_generated: dict[str, dict[str, str]] = {}

    for half_life, (vectors, metadata) in results.items():
        vector_path = output_dir / f"user_vectors_halflife_{half_life}.npz"
        metadata_path = output_dir / f"user_metadata_halflife_{half_life}.json"
        stats_path = output_dir / f"summary_stats_halflife_{half_life}.txt"

        np.savez_compressed(vector_path, vectors=vectors)
        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        norms = np.linalg.norm(vectors, axis=1)
        stats_path.write_text(
            "\n".join(
                [
                    f"Half-life: {half_life} days",
                    f"Number of users: {len(vectors)}",
                    f"Vector dimension: {vectors.shape[1]}",
                    "",
                    "Vector L2 Norm Statistics:",
                    f"  min    : {norms.min():.6f}",
                    f"  max    : {norms.max():.6f}",
                    f"  mean   : {norms.mean():.6f}",
                    f"  median : {np.median(norms):.6f}",
                    f"  std    : {norms.std():.6f}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        files_generated[str(half_life)] = {
            "vectors": f"{output_dir.name}/{vector_path.name}",
            "metadata": f"{output_dir.name}/{metadata_path.name}",
            "stats": f"{output_dir.name}/{stats_path.name}",
        }

    summary = {
        "description": "Temporal-weighted user profile vectors",
        "formula": "weight(p) = exp(-log(2) * days_since_read(p) / half_life)",
        "user_vector": "weighted_mean(embeddings, weights)",
        "paper_index_mapping": "arxiv_dataset/train.csv row order",
        "half_lives_evaluated": list(results),
        "total_users": len(next(iter(results.values()))[1]),
        matrix_name: list(matrix_shape),
        "files_generated": files_generated,
    }
    (output_dir / "README.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate correctly aligned TF-IDF and SBERT user profiles."
    )
    parser.add_argument(
        "--model",
        choices=("tfidf", "sbert", "all"),
        default="all",
    )
    args = parser.parse_args()

    df = pd.read_csv(PROJECT_ROOT / "arxiv_dataset/train.csv")
    paper_ids = df["paper_id"].tolist()
    with (PROJECT_ROOT / "UserDataGenerator/synthetic_users.json").open(
        encoding="utf-8"
    ) as file:
        users = json.load(file)["train"]

    jobs: list[tuple[str, Any, Path, str]] = []
    if args.model in ("tfidf", "all"):
        jobs.append(
            (
                "TF-IDF",
                scipy.sparse.load_npz(
                    PROJECT_ROOT / "arxiv_dataset/tfidf_matrix.npz"
                ),
                PROJECT_ROOT / "user_profiles",
                "tfidf_matrix_shape",
            )
        )
    if args.model in ("sbert", "all"):
        jobs.append(
            (
                "SBERT",
                np.load(
                    PROJECT_ROOT / "arxiv_dataset/SBERTEmbeddings_final.npy",
                    mmap_mode="r",
                ),
                PROJECT_ROOT / "user_profiles_SBERT",
                "SBERT_matrix_shape",
            )
        )

    for label, embeddings, output_dir, matrix_name in jobs:
        print(f"Generating {label} profiles...")
        results = generate_profiles(users, embeddings, paper_ids)
        save_profiles(output_dir, results, matrix_name, embeddings.shape)
        print(f"Saved {label} profiles to {output_dir}")


if __name__ == "__main__":
    main()
