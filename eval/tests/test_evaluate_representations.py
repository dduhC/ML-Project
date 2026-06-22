import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse


EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from evaluate_representations import (  # noqa: E402
    Representation,
    build_user_profile,
    category_precision_at_k,
    evaluate_representation,
    intra_list_diversity,
    rank_candidates,
    semantic_coherence_at_k,
    summarize_metrics,
)


class MetricTests(unittest.TestCase):
    def test_category_precision_matches_known_value(self):
        ranked = np.array([4, 2, 3, 0, 1])
        categories = np.array(["A", "B", "A", "C", "A"])

        self.assertAlmostEqual(
            category_precision_at_k(ranked, categories, {"A"}, k=3),
            2 / 3,
        )

    def test_temporal_profile_matches_for_dense_and_sparse(self):
        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        dense = build_user_profile(matrix, [0, 1], [30, 0], half_life=30)
        sparse = build_user_profile(
            scipy.sparse.csr_matrix(matrix),
            [0, 1],
            [30, 0],
            half_life=30,
        )

        np.testing.assert_allclose(dense, [[1 / 3, 2 / 3]])
        np.testing.assert_allclose(sparse.toarray(), dense)

    def test_ranking_excludes_history_and_uses_hybrid_score(self):
        matrix = np.eye(3)
        profile = np.array([[1.0, 0.0, 0.0]])
        ranked, similarities = rank_candidates(
            profile,
            matrix,
            recency_scores=np.array([0.1, 1.0, 0.0]),
            excluded_indices=[0],
            alpha=0.7,
        )

        self.assertNotEqual(ranked[0], 0)
        self.assertEqual(ranked[0], 1)
        np.testing.assert_allclose(similarities, [1.0, 0.0, 0.0])

    def test_coherence_and_ild_match_known_values(self):
        ranked = np.array([0, 1])
        similarities = np.array([0.8, 0.4])
        matrix = np.eye(2)

        self.assertAlmostEqual(
            semantic_coherence_at_k(similarities, ranked, 2), 0.6
        )
        self.assertAlmostEqual(intra_list_diversity(matrix, ranked, 2), 1.0)


class EvaluationSmokeTests(unittest.TestCase):
    def test_three_representations_share_users_and_produce_summary(self):
        df = pd.DataFrame(
            {
                "paper_id": ["p0", "p1", "p2", "p3", "p4", "p5"],
                "primary_category": ["A", "A", "A", "B", "B", "B"],
                "recency_score": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            }
        )
        users = [
            {
                "user_id": "u0",
                "archetype": "test",
                "user_type": "new_user",
                "train_history": [["p0", 30], ["p1", 0]],
                "target_paper": "p2",
                "negative_papers": ["p4"],
            }
        ]
        matrices = [
            Representation("TF-IDF", scipy.sparse.csr_matrix(np.eye(6))),
            Representation("SBERT pretrained", np.eye(6)),
            Representation("SBERT fine-tuned", np.eye(6)),
        ]
        mapping = {paper_id: index for index, paper_id in enumerate(df.paper_id)}
        rows = []
        for representation in matrices:
            rows.extend(
                evaluate_representation(
                    representation,
                    users,
                    mapping,
                    df.primary_category.to_numpy(),
                    df.recency_score.to_numpy(),
                    [2],
                    alpha=0.7,
                    half_life=90,
                )
            )

        per_user = pd.DataFrame(rows)
        summary = summarize_metrics(per_user)
        self.assertEqual(
            set(summary["representation"]),
            {"TF-IDF", "SBERT pretrained", "SBERT fine-tuned"},
        )
        self.assertEqual(set(per_user["user_id"]), {"u0"})
        self.assertEqual(set(per_user["k"]), {2})


if __name__ == "__main__":
    unittest.main()
