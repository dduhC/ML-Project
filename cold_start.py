from collections.abc import Sequence

import numpy as np
import pandas as pd


def build_cold_start_profile(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    selected_categories: Sequence[str],
    papers_per_category: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a cold-start profile from recent papers in each selected category."""
    categories = list(dict.fromkeys(selected_categories))

    if not categories:
        raise ValueError("Select at least one category.")
    if papers_per_category <= 0:
        raise ValueError("papers_per_category must be greater than zero.")
    if len(df) != len(embeddings):
        raise ValueError(
            "DataFrame and embeddings must contain the same number of papers."
        )

    required_columns = {"primary_category", "recency_score"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}.")

    selected_indices: list[int] = []
    category_values = df["primary_category"].to_numpy()
    recency_values = df["recency_score"].to_numpy()

    for category in categories:
        category_indices = np.flatnonzero(category_values == category)
        if category_indices.size == 0:
            raise ValueError(f"No papers found for category '{category}'.")

        category_recency = recency_values[category_indices]
        if pd.isna(category_recency).any():
            raise ValueError(
                f"Category '{category}' contains papers without recency scores."
            )

        recent_order = np.argsort(-category_recency, kind="stable")
        selected_indices.extend(
            category_indices[recent_order[:papers_per_category]].tolist()
        )

    seed_indices = np.asarray(selected_indices, dtype=np.int64)
    seed_embeds = embeddings[seed_indices]
    import scipy.sparse
    if scipy.sparse.issparse(seed_embeds):
        seed_embeds = seed_embeds.toarray()
    profile_vector = np.asarray(seed_embeds).mean(axis=0).flatten()
    return profile_vector, seed_indices
