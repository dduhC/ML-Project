"""Compare pretrained and fine-tuned SBERT embeddings with t-SNE."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "arxiv_dataset"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "tsne_compare_sbert.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a side-by-side t-SNE comparison of SBERT embeddings."
    )
    parser.add_argument(
        "--before",
        type=Path,
        default=DEFAULT_DATA_DIR / "SBERTEmbeddings_test.npy",
        help="Pretrained SBERT embedding file.",
    )
    parser.add_argument(
        "--after",
        type=Path,
        default=DEFAULT_DATA_DIR / "SBERTEmbeddings_test_final.npy",
        help="Fine-tuned SBERT embedding file.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_DATA_DIR / "test.csv",
        help="CSV containing primary_category in embedding row order.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination PNG path.",
    )
    parser.add_argument("--top-categories", type=int, default=8)
    parser.add_argument("--samples-per-category", type=int, default=300)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.before, args.after, args.metadata):
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")

    if args.top_categories < 2:
        raise ValueError("--top-categories must be at least 2")
    if args.samples_per_category < 2:
        raise ValueError("--samples-per-category must be at least 2")
    if args.perplexity <= 0:
        raise ValueError("--perplexity must be positive")


def load_inputs(
    before_path: Path, after_path: Path, metadata_path: Path
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    before = np.load(before_path)
    after = np.load(after_path)
    metadata = pd.read_csv(metadata_path, usecols=["primary_category"])
    labels = metadata["primary_category"]

    if before.ndim != 2 or after.ndim != 2:
        raise ValueError("Both embedding arrays must be two-dimensional")
    if before.shape != after.shape:
        raise ValueError(
            f"Embedding shape mismatch: before={before.shape}, after={after.shape}"
        )
    if len(labels) != before.shape[0]:
        raise ValueError(
            "Metadata/embedding row mismatch: "
            f"metadata={len(labels)}, embeddings={before.shape[0]}"
        )
    if labels.isna().any():
        raise ValueError("primary_category contains missing values")
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("Embedding arrays contain NaN or infinite values")

    return before, after, labels


def balanced_sample_indices(
    labels: pd.Series,
    top_categories: int,
    samples_per_category: int,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    categories = labels.value_counts().head(top_categories).index.tolist()
    if len(categories) < 2:
        raise ValueError("At least two categories are required for comparison")

    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for category in categories:
        candidates = np.flatnonzero(labels.to_numpy() == category)
        sample_size = min(samples_per_category, len(candidates))
        selected.append(rng.choice(candidates, size=sample_size, replace=False))

    indices = np.concatenate(selected)
    rng.shuffle(indices)
    return indices, categories


def reduce_with_tsne(
    embeddings: np.ndarray, perplexity: float, seed: int
) -> np.ndarray:
    if perplexity >= len(embeddings):
        raise ValueError(
            f"Perplexity ({perplexity}) must be smaller than sample count "
            f"({len(embeddings)})"
        )

    pca_components = min(50, embeddings.shape[1], len(embeddings) - 1)
    reduced = PCA(n_components=pca_components, random_state=seed).fit_transform(
        embeddings
    )
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(reduced)


def compute_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    return float(silhouette_score(embeddings, labels, metric="cosine"))


def plot_comparison(
    before_2d: np.ndarray,
    after_2d: np.ndarray,
    labels: np.ndarray,
    categories: list[str],
    before_score: float,
    after_score: float,
    output_path: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    palette = dict(
        zip(categories, sns.color_palette("tab10", n_colors=len(categories)))
    )
    figure, axes = plt.subplots(1, 2, figsize=(18, 8))

    panels = (
        (axes[0], before_2d, "Pretrained SBERT", before_score),
        (axes[1], after_2d, "Fine-tuned SBERT", after_score),
    )
    for axis, coordinates, title, score in panels:
        frame = pd.DataFrame(
            {
                "t-SNE 1": coordinates[:, 0],
                "t-SNE 2": coordinates[:, 1],
                "Category": labels,
            }
        )
        sns.scatterplot(
            data=frame,
            x="t-SNE 1",
            y="t-SNE 2",
            hue="Category",
            hue_order=categories,
            palette=palette,
            s=18,
            alpha=0.72,
            linewidth=0,
            ax=axis,
            legend=False,
        )
        axis.set_title(f"{title}\nSilhouette score: {score:.4f}")
        axis.set_xlabel("t-SNE dimension 1")
        axis.set_ylabel("t-SNE dimension 2")

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=palette[category],
            markeredgecolor="none",
            markersize=7,
            label=category,
        )
        for category in categories
    ]
    figure.legend(
        handles=handles,
        labels=categories,
        title="Primary category",
        loc="lower center",
        ncol=min(4, len(categories)),
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.suptitle(
        "t-SNE Comparison: SBERT Before vs. After Fine-tuning",
        fontsize=16,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.94))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    validate_args(args)

    before, after, all_labels = load_inputs(
        args.before, args.after, args.metadata
    )
    indices, categories = balanced_sample_indices(
        all_labels,
        args.top_categories,
        args.samples_per_category,
        args.seed,
    )
    labels = all_labels.iloc[indices].to_numpy()
    before_sample = before[indices]
    after_sample = after[indices]

    print(
        f"Using {len(indices)} papers across {len(categories)} categories: "
        f"{', '.join(categories)}"
    )
    print("Computing pretrained SBERT t-SNE...")
    before_2d = reduce_with_tsne(before_sample, args.perplexity, args.seed)
    print("Computing fine-tuned SBERT t-SNE...")
    after_2d = reduce_with_tsne(after_sample, args.perplexity, args.seed)

    before_score = compute_silhouette(before_sample, labels)
    after_score = compute_silhouette(after_sample, labels)
    plot_comparison(
        before_2d,
        after_2d,
        labels,
        categories,
        before_score,
        after_score,
        args.output,
    )

    print(f"Pretrained silhouette score: {before_score:.4f}")
    print(f"Fine-tuned silhouette score: {after_score:.4f}")
    print(f"Saved comparison image to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
