"""Evaluate TF-IDF, pretrained SBERT and fine-tuned SBERT on the test set."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
DATA_DIR = ROOT / "arxiv_dataset"
OUTPUT_DIR = EVAL_DIR / "outputs"

N_USERS = 500
SEED = 42
K_VALUES = (5, 10, 20)
ALPHA = 0.7
HALF_LIFE = 90

if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from generate_test_users import (  # noqa: E402
    generate_synthetic_test_users,
    save_users,
    validate_users,
)


@dataclass(frozen=True)
class Representation:
    name: str
    matrix: np.ndarray | scipy.sparse.spmatrix


def load_representations() -> list[Representation]:
    return [
        Representation(
            "TF-IDF",
            scipy.sparse.load_npz(DATA_DIR / "tfidf_test_matrix.npz"),
        ),
        Representation(
            "SBERT pretrained",
            np.load(DATA_DIR / "SBERTEmbeddings_test.npy", mmap_mode="r"),
        ),
        Representation(
            "SBERT fine-tuned",
            np.load(DATA_DIR / "SBERTEmbeddings_test_final.npy", mmap_mode="r"),
        ),
    ]


def temporal_weights(days_ago: Sequence[float], half_life: float) -> np.ndarray:
    days = np.asarray(days_ago, dtype=float)
    weights = np.exp(-np.log(2) * days / half_life)
    return weights / weights.sum()


def build_user_profile(
    matrix: np.ndarray | scipy.sparse.spmatrix,
    history_indices: Sequence[int],
    days_ago: Sequence[float],
    half_life: float,
) -> np.ndarray | scipy.sparse.csr_matrix:
    weights = temporal_weights(days_ago, half_life)
    history = matrix[list(history_indices)]
    if scipy.sparse.issparse(history):
        return scipy.sparse.csr_matrix(
            history.multiply(weights[:, None]).sum(axis=0)
        )
    return np.average(history, axis=0, weights=weights).reshape(1, -1)


def rank_candidates(
    profile: np.ndarray | scipy.sparse.spmatrix,
    matrix: np.ndarray | scipy.sparse.spmatrix,
    recency_scores: np.ndarray,
    excluded_indices: Sequence[int],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    similarities = cosine_similarity(profile, matrix).ravel()
    scores = alpha * similarities + (1 - alpha) * recency_scores
    scores[list(excluded_indices)] = -np.inf
    return np.argsort(scores, kind="stable")[::-1], similarities


def category_precision_at_k(
    ranked_indices: Sequence[int],
    categories: np.ndarray,
    relevant_categories: set[str],
    k: int,
) -> float:
    return float(
        np.isin(categories[np.asarray(ranked_indices[:k])], list(relevant_categories))
        .mean()
    )


def semantic_coherence_at_k(
    similarities: np.ndarray, ranked_indices: Sequence[int], k: int
) -> float:
    return float(np.mean(similarities[np.asarray(ranked_indices[:k])]))


def intra_list_diversity(
    matrix: np.ndarray | scipy.sparse.spmatrix,
    ranked_indices: Sequence[int],
    k: int,
) -> float:
    indices = np.asarray(ranked_indices[:k])
    if len(indices) < 2:
        return 0.0
    pairwise = cosine_similarity(matrix[indices])
    return float(np.mean(1 - pairwise[np.triu_indices(len(indices), k=1)]))


def evaluate_representation(
    representation: Representation,
    users: Sequence[dict[str, Any]],
    paper_id_to_index: dict[str, int],
    categories: np.ndarray,
    recency_scores: np.ndarray,
    k_values: Sequence[int],
    alpha: float,
    half_life: float,
) -> list[dict[str, Any]]:
    rows = []
    for user in users:
        history_indices = [
            paper_id_to_index[paper_id]
            for paper_id, _ in user["train_history"]
        ]
        days_ago = [days for _, days in user["train_history"]]
        relevant_categories = set(categories[history_indices])

        started = time.perf_counter()
        profile = build_user_profile(
            representation.matrix, history_indices, days_ago, half_life
        )
        profile_seconds = time.perf_counter() - started

        started = time.perf_counter()
        ranked, similarities = rank_candidates(
            profile,
            representation.matrix,
            recency_scores,
            history_indices,
            alpha,
        )
        ranking_seconds = time.perf_counter() - started

        for k in k_values:
            rows.append(
                {
                    "representation": representation.name,
                    "user_id": user["user_id"],
                    "archetype": user["archetype"],
                    "user_type": user["user_type"],
                    "history_size": len(history_indices),
                    "k": k,
                    "category_precision": category_precision_at_k(
                        ranked, categories, relevant_categories, k
                    ),
                    "semantic_coherence": semantic_coherence_at_k(
                        similarities, ranked, k
                    ),
                    "ild": intra_list_diversity(
                        representation.matrix, ranked, k
                    ),
                    "profile_seconds": profile_seconds,
                    "ranking_seconds": ranking_seconds,
                }
            )
    return rows


def summarize_metrics(per_user: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "category_precision",
        "semantic_coherence",
        "ild",
        "profile_seconds",
        "ranking_seconds",
    ]
    summary = (
        per_user.groupby(["representation", "k"], sort=False)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary


def load_users(df: pd.DataFrame) -> list[dict[str, Any]]:
    path = OUTPUT_DIR / "synthetic_test_users.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            len(payload.get("test", [])) == N_USERS
            and payload.get("meta", {}).get("random_seed") == SEED
        ):
            return payload["test"]

    users = generate_synthetic_test_users(df, n_users=N_USERS, seed=SEED)
    validate_users(users, df)
    save_users(users, path, seed=SEED, n_negatives=6)
    return users


def create_metrics_chart(
    summary: pd.DataFrame,
    silhouette_scores: dict[str, float],
) -> None:
    plots = [
        ("category_precision_mean", "Category Precision@K"),
        ("semantic_coherence_mean", "Semantic Coherence"),
        ("ild_mean", "Intra-list Diversity"),
    ]
    models = summary["representation"].unique()
    k_values = sorted(summary["k"].unique())
    x = np.arange(len(k_values))
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(models)]

    for axis, (column, title) in zip(axes, plots):
        for offset, model in enumerate(models):
            subset = summary[summary["representation"] == model].set_index("k")
            axis.bar(
                x + (offset - 1) * 0.24,
                [subset.loc[k, column] for k in k_values],
                0.24,
                label=model,
                color=colors[offset],
            )
        axis.set_title(title)
        axis.set_xticks(x, [f"K={k}" for k in k_values])
        axis.grid(axis="y", alpha=0.25)

    silhouette_axis = axes[3]
    silhouette_values = [silhouette_scores[model] for model in models]
    silhouette_axis.bar(
        np.arange(len(models)),
        silhouette_values,
        color=colors,
    )
    silhouette_axis.set_title("Silhouette Score")
    silhouette_axis.set_xticks(
        np.arange(len(models)),
        [model.replace(" ", "\n", 1) for model in models],
    )
    silhouette_axis.grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(
        OUTPUT_DIR / "representation_metrics.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(figure)


def create_tsne(
    representations: Sequence[Representation], categories: np.ndarray
) -> dict[str, float]:
    top_categories = pd.Series(categories).value_counts().head(8).index
    rng = np.random.default_rng(SEED)
    indices = np.concatenate(
        [
            rng.choice(
                np.flatnonzero(categories == category),
                size=min(200, np.sum(categories == category)),
                replace=False,
            )
            for category in top_categories
        ]
    )
    rng.shuffle(indices)
    labels = categories[indices]

    figure, axes = plt.subplots(1, 3, figsize=(21, 7))
    scores = {}
    for axis, representation in zip(axes, representations):
        sample = representation.matrix[indices]
        scores[representation.name] = float(
            silhouette_score(sample, labels, metric="cosine")
        )
        components = min(50, sample.shape[0] - 1, sample.shape[1] - 1)
        reducer = (
            TruncatedSVD(components, random_state=SEED)
            if scipy.sparse.issparse(sample)
            else PCA(components, random_state=SEED)
        )
        coordinates = TSNE(
            perplexity=30,
            init="pca",
            learning_rate="auto",
            random_state=SEED,
        ).fit_transform(reducer.fit_transform(sample))

        for category in top_categories:
            mask = labels == category
            axis.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=12,
                alpha=0.68,
                label=category,
            )
        axis.set_title(
            f"{representation.name}\nSilhouette: "
            f"{scores[representation.name]:.4f}"
        )

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4)
    figure.tight_layout(rect=(0, 0.10, 1, 1))
    figure.savefig(
        OUTPUT_DIR / "tsne_compare_all.png", dpi=300, bbox_inches="tight"
    )
    plt.close(figure)
    return scores


def markdown_table(frame: pd.DataFrame) -> list[str]:
    headers = frame.columns.tolist()
    rows = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    rows.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return rows


def write_outputs(
    per_user: pd.DataFrame,
    summary: pd.DataFrame,
    silhouette_scores: dict[str, float],
) -> None:
    per_user.to_csv(OUTPUT_DIR / "per_user_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "representation_metrics.csv", index=False)

    payload = {
        "config": {
            "n_users": N_USERS,
            "seed": SEED,
            "k": list(K_VALUES),
            "alpha": ALPHA,
            "half_life": HALF_LIFE,
        },
        "silhouette_scores": silhouette_scores,
        "metrics": summary.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "representation_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    columns = [
        "representation",
        "k",
        "category_precision_mean",
        "semantic_coherence_mean",
        "ild_mean",
    ]
    table = summary[columns].round(4)
    lines = [
        "# Representation Evaluation Summary",
        "",
        (
            f"Protocol: {N_USERS} synthetic test users, hybrid alpha={ALPHA}, "
            f"profile half-life={HALF_LIFE} days, seed={SEED}."
        ),
        "",
        *markdown_table(table),
        "",
        "## Silhouette scores",
        "",
        "| Representation | Silhouette (cosine) |",
        "|---|---:|",
        *[
            f"| {name} | {score:.4f} |"
            for name, score in silhouette_scores.items()
        ],
        "",
        (
            "t-SNE is a qualitative visualization. Ranking metrics are the "
            "primary evidence for recommendation quality."
        ),
    ]
    (OUTPUT_DIR / "evaluation_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_DIR / "test.csv")
    representations = load_representations()
    users = load_users(df)
    paper_id_to_index = {
        str(paper_id): index for index, paper_id in enumerate(df["paper_id"])
    }
    categories = df["primary_category"].astype(str).to_numpy()
    recency_scores = df["recency_score"].to_numpy(float)

    rows = []
    for representation in representations:
        print(f"Evaluating {representation.name}...")
        rows.extend(
            evaluate_representation(
                representation,
                users,
                paper_id_to_index,
                categories,
                recency_scores,
                K_VALUES,
                ALPHA,
                HALF_LIFE,
            )
        )

    per_user = pd.DataFrame(rows)
    summary = summarize_metrics(per_user)
    silhouette_scores = create_tsne(representations, categories)
    create_metrics_chart(summary, silhouette_scores)
    write_outputs(per_user, summary, silhouette_scores)
    print(f"Evaluation outputs: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
