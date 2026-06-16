import random
import gc
from sentence_transformers import InputExample
import numpy as np

def sample_anchors_by_category(df, n_anchors):
    """
    Select anchor documents using stratified sampling.
    Each category gets a number of anchors proportional to its size in the dataset.
    Minimum 10 anchors per category to avoid dropping rare categories.

    Returns: list of row indices (in df) selected as anchors.
    """

    # Count documents per category and compute proportions
    category_counts      = df['primary_category'].value_counts()
    category_proportions = category_counts / len(df)

    # Convert proportions to quota per category, with a floor of 10
    category_quotas = (category_proportions * n_anchors).astype(int)
    category_quotas = category_quotas.clip(lower=10)

    # Randomly sample each category according to its quota
    selected_indices = []
    for category_name, quota in category_quotas.items():
        all_indices_in_category = df[df['primary_category'] == category_name].index.tolist()
        actual_quota            = min(len(all_indices_in_category), quota)
        sampled                 = random.sample(all_indices_in_category, actual_quota)
        selected_indices.extend(sampled)

    # Shuffle so batches aren't biased toward any one category
    random.shuffle(selected_indices)

    return selected_indices


def get_text(df, idx):
    """Concatenate title + abstract of a document into a single string."""
    row = df.iloc[idx]
    return str(row['title']) + " || " + str(row['abstract'])


def find_top100_similar(batch_tfidf, full_tfidf_matrix):
    """
    Find the Top-100 most similar documents for each anchor in the batch
    by multiplying their TF-IDF vectors against the full matrix.

    Returns: list of numpy arrays, each containing 100 global indices
             sorted from most similar → least similar.
    """
    # Matrix multiply: (batch_size × 5000) × (5000 × 136k) → (batch_size × 136k)
    # Each cell [i][j] = cosine similarity between anchor_i and document_j
    sim_matrix = batch_tfidf.dot(full_tfidf_matrix.T).toarray()

    top100_per_anchor = []
    for sim_scores in sim_matrix:
        # argpartition is O(n) — guarantees top-100 are at the end, but unsorted within that group
        top100_unsorted = np.argpartition(sim_scores, -100)[-100:]

        # Sort those 100 in descending order of similarity
        sort_order      = np.argsort(sim_scores[top100_unsorted])[::-1]
        top100_sorted   = top100_unsorted[sort_order]

        top100_per_anchor.append(top100_sorted)

    return top100_per_anchor


def build_semantic_triplet(df, anchor_idx, top100_indices):
    """
    Build Triplet 1 — Semantic Hard Negative:
      - Anchor:        the source document
      - Positive:      the most similar document (rank #1 in Top-100)
      - Hard Negative: a document that is fairly similar but not the most relevant (rank #50–99)
                       → harder to distinguish → forces the model to learn finer-grained differences

    Returns an InputExample, or None if there aren't enough candidates.
    """
    # Remove the anchor itself from the candidate list
    top100 = top100_indices[top100_indices != anchor_idx]

    if len(top100) < 2:
        return None

    anchor_text   = get_text(df, anchor_idx)
    positive_text = get_text(df, top100[0])       # Most similar document

    hard_neg_pool = top100[50:100]                 # Ranks 50–99: similar but not the closest
    if len(hard_neg_pool) == 0:
        hard_neg_pool = top100[1:]                 # Fallback if fewer than 100 results
    hard_neg_text = get_text(df, random.choice(hard_neg_pool))

    return InputExample(texts=[anchor_text, positive_text, hard_neg_text])


def build_category_triplet(df, anchor_idx, anchor_category, category_to_indices, all_categories):
    """
    Build Triplet 2 — Category Easy Negative:
      - Anchor:        the source document
      - Positive:      a random document from the same category (same topic)
      - Easy Negative: a random document from a completely different category
                       → easy to distinguish → teaches the model broad topic signals

    Returns an InputExample, or None if there are no same-category documents available.
    """
    # Gather all documents in the same category, excluding the anchor itself
    same_category_indices = list(category_to_indices[anchor_category])
    if anchor_idx in same_category_indices:
        same_category_indices.remove(anchor_idx)

    if not same_category_indices:
        return None

    anchor_text   = get_text(df, anchor_idx)
    positive_text = get_text(df, random.choice(same_category_indices))

    # Pick a random document from a different category
    other_categories = [c for c in all_categories if c != anchor_category]
    random_category  = random.choice(other_categories)
    easy_neg_text    = get_text(df, random.choice(list(category_to_indices[random_category])))

    return InputExample(texts=[anchor_text, positive_text, easy_neg_text])


def generate_triplet_dataset(df, tfidf_matrix, n_anchors=30000, batch_size=1000):
    """
    Main function: generate a triplet dataset for training an embedding model.

    Strategy:
      1. Select 30,000 anchors via stratified sampling by category
      2. Process in batches of 1,000 anchors to avoid running out of RAM
      3. For each anchor, create 2 triplets:
           - Semantic triplet (Hard Negative): teaches fine-grained content similarity
           - Category triplet (Easy Negative): teaches broad topic discrimination

    Args:
      df           -- DataFrame with ~136k documents (needs: title, abstract, primary_category)
      tfidf_matrix -- Sparse TF-IDF matrix of shape (136k, 5000)
      n_anchors    -- Number of anchors to sample (default: 30,000)
      batch_size   -- Number of anchors to process per batch (default: 1,000)

    Returns:
      List of ~60,000 InputExample objects (~2 triplets per anchor)
    """
    # --- Setup ---
    anchor_indices   = sample_anchors_by_category(df, n_anchors)
    total_anchors    = len(anchor_indices)
    category_to_idx  = df.groupby('primary_category').groups  # {category: [idx, ...]}
    all_categories   = list(category_to_idx.keys())
    all_triplets     = []

    # --- Process in batches ---
    for batch_start in range(0, total_anchors, batch_size):
        batch_end = min(batch_start + batch_size, total_anchors)

        batch_anchor_indices = anchor_indices[batch_start:batch_end]
        batch_tfidf          = tfidf_matrix[batch_anchor_indices]

        # Compute Top-100 similar documents for all anchors in the batch at once
        top100_per_anchor = find_top100_similar(batch_tfidf, tfidf_matrix)

        # Build 2 triplets per anchor
        for local_i, anchor_idx in enumerate(batch_anchor_indices):
            anchor_category = df.iloc[anchor_idx]['primary_category']
            top100          = top100_per_anchor[local_i]

            semantic_triplet = build_semantic_triplet(df, anchor_idx, top100)
            category_triplet = build_category_triplet(df, anchor_idx, anchor_category,
                                                       category_to_idx, all_categories)

            if semantic_triplet:
                all_triplets.append(semantic_triplet)
            if category_triplet:
                all_triplets.append(category_triplet)

        # Free RAM after each batch (~500MB+ per batch)
        del batch_tfidf, top100_per_anchor
        gc.collect()
    return all_triplets