from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize


def prepare_user_tsne_data(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    user_vector: np.ndarray,
    recommendation_indices: Sequence[int],
    context_indices: Sequence[int],
    context_label: str,
    samples_per_category: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Prepare balanced background and highlighted vectors for user t-SNE."""
    _validate_inputs(
        df,
        embeddings,
        user_vector,
        recommendation_indices,
        context_indices,
        samples_per_category,
    )

    recommendation_indices = _unique_indices(recommendation_indices)
    recommendation_set = set(recommendation_indices)
    context_indices = [
        idx for idx in _unique_indices(context_indices)
        if idx not in recommendation_set
    ]
    highlight_indices = recommendation_indices + context_indices

    categories = (
        df.iloc[highlight_indices]["primary_category"]
        .value_counts()
        .index.tolist()
    )
    if not categories:
        raise ValueError("At least one context or recommendation paper is required.")

    rng = np.random.default_rng(seed)
    excluded = set(highlight_indices)
    category_values = df["primary_category"].to_numpy()
    background_indices: list[int] = []
    for category in categories:
        candidates = np.flatnonzero(category_values == category)
        candidates = np.asarray(
            [idx for idx in candidates if idx not in excluded], dtype=np.int64
        )
        sample_size = min(samples_per_category, len(candidates))
        if sample_size:
            sampled = rng.choice(candidates, size=sample_size, replace=False)
            background_indices.extend(sampled.tolist())

    rows: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []
    for idx in background_indices:
        rows.append(
            {
                "paper_index": idx,
                "category": category_values[idx],
                "role": "Background",
                "rank": np.nan,
            }
        )
        vectors.append(embeddings[idx])

    for idx in context_indices:
        rows.append(
            {
                "paper_index": idx,
                "category": category_values[idx],
                "role": context_label,
                "rank": np.nan,
            }
        )
        vectors.append(embeddings[idx])

    for rank, idx in enumerate(recommendation_indices, start=1):
        rows.append(
            {
                "paper_index": idx,
                "category": category_values[idx],
                "role": "Recommendation",
                "rank": rank,
            }
        )
        vectors.append(embeddings[idx])

    rows.append(
        {
            "paper_index": -1,
            "category": "User profile",
            "role": "User profile",
            "rank": np.nan,
        }
    )
    vectors.append(user_vector)

    frame = pd.DataFrame(rows)
    frame.attrs["vectors"] = normalize(np.asarray(vectors), norm="l2")
    frame.attrs["categories"] = categories
    return frame


def reduce_user_tsne(
    vectors: np.ndarray,
    perplexity: float = 30.0,
    seed: int = 42,
) -> np.ndarray:
    """Reduce normalized embedding vectors to two dimensions."""
    if vectors.ndim != 2 or len(vectors) < 3:
        raise ValueError("t-SNE requires at least three two-dimensional vectors.")
    if not np.isfinite(vectors).all():
        raise ValueError("t-SNE vectors contain NaN or infinite values.")

    adjusted_perplexity = min(perplexity, max(1.0, (len(vectors) - 1) / 3))
    pca_components = min(50, vectors.shape[1], len(vectors) - 1)
    reduced = PCA(
        n_components=pca_components,
        random_state=seed,
    ).fit_transform(vectors)
    return TSNE(
        n_components=2,
        perplexity=adjusted_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(reduced)


def create_user_profile_tsne(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    user_vector: np.ndarray,
    recommendation_indices: Sequence[int],
    context_indices: Sequence[int],
    context_label: str,
    samples_per_category: int = 100,
    perplexity: float = 30.0,
    seed: int = 42,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Create a t-SNE figure for a user profile and its recommendations."""
    frame = prepare_user_tsne_data(
        df=df,
        embeddings=embeddings,
        user_vector=user_vector,
        recommendation_indices=recommendation_indices,
        context_indices=context_indices,
        context_label=context_label,
        samples_per_category=samples_per_category,
        seed=seed,
    )
    coordinates = reduce_user_tsne(
        frame.attrs["vectors"],
        perplexity=perplexity,
        seed=seed,
    )
    frame["tsne_1"] = coordinates[:, 0]
    frame["tsne_2"] = coordinates[:, 1]

    figure, axis = plt.subplots(figsize=(11, 8))
    color_map = plt.get_cmap("tab10")
    categories = frame.attrs["categories"]
    for category_index, category in enumerate(categories):
        points = frame[
            (frame["role"] == "Background") & (frame["category"] == category)
        ]
        axis.scatter(
            points["tsne_1"],
            points["tsne_2"],
            s=18,
            alpha=0.35,
            color=color_map(category_index % 10),
            label=category,
        )

    context = frame[frame["role"] == context_label]
    if not context.empty:
        axis.scatter(
            context["tsne_1"],
            context["tsne_2"],
            marker="x",
            s=55,
            linewidths=1.5,
            color="black",
            label=context_label,
        )

    recommendations = frame[frame["role"] == "Recommendation"]
    axis.scatter(
        recommendations["tsne_1"],
        recommendations["tsne_2"],
        s=90,
        facecolors="none",
        edgecolors="darkorange",
        linewidths=2,
        label="Top-K recommendations",
    )
    for row in recommendations.itertuples():
        axis.annotate(
            str(int(row.rank)),
            (row.tsne_1, row.tsne_2),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )

    user = frame[frame["role"] == "User profile"].iloc[0]
    axis.scatter(
        [user["tsne_1"]],
        [user["tsne_2"]],
        marker="*",
        s=300,
        color="red",
        edgecolors="black",
        linewidths=0.8,
        label="User profile",
        zorder=5,
    )

    axis.set_title("User Profile and Top-K Recommendations in t-SNE Space")
    axis.set_xlabel("t-SNE dimension 1")
    axis.set_ylabel("t-SNE dimension 2")
    axis.grid(alpha=0.2)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=300, bbox_inches="tight")

    return figure


def _validate_inputs(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    user_vector: np.ndarray,
    recommendation_indices: Sequence[int],
    context_indices: Sequence[int],
    samples_per_category: int,
) -> None:
    if "primary_category" not in df.columns:
        raise ValueError("DataFrame must contain primary_category.")
    if embeddings.ndim != 2 or len(df) != len(embeddings):
        raise ValueError("DataFrame and embeddings must have matching paper rows.")
    user_vector = np.asarray(user_vector)
    if user_vector.ndim != 1 or user_vector.shape[0] != embeddings.shape[1]:
        raise ValueError("User vector dimension must match paper embeddings.")
    if not np.isfinite(embeddings).all() or not np.isfinite(user_vector).all():
        raise ValueError("Embeddings and user vector must contain finite values.")
    if samples_per_category <= 0:
        raise ValueError("samples_per_category must be greater than zero.")

    all_indices = list(recommendation_indices) + list(context_indices)
    if any(idx < 0 or idx >= len(df) for idx in all_indices):
        raise ValueError("Paper index is outside the dataset.")


def _unique_indices(indices: Sequence[int]) -> list[int]:
    return list(dict.fromkeys(int(idx) for idx in indices))
