import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cold_start import build_cold_start_profile


class BuildColdStartProfileTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "primary_category": ["cs.CV"] * 25 + ["cs.LG"] * 22 + ["cs.CL"] * 3,
                "recency_score": (
                    list(range(25))
                    + list(range(100, 122))
                    + [9.0, 8.0, 7.0]
                ),
            }
        )
        self.embeddings = np.arange(len(self.df) * 4, dtype=float).reshape(-1, 4)

    def test_selects_twenty_most_recent_papers_per_category(self):
        profile, indices = build_cold_start_profile(
            self.df, self.embeddings, ["cs.CV", "cs.LG"]
        )

        expected_indices = np.array(
            list(range(24, 4, -1)) + list(range(46, 26, -1))
        )
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_allclose(
            profile, self.embeddings[expected_indices].mean(axis=0)
        )

    def test_uses_all_papers_when_category_has_fewer_than_twenty(self):
        profile, indices = build_cold_start_profile(
            self.df, self.embeddings, ["cs.CL"]
        )

        expected_indices = np.array([47, 48, 49])
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_allclose(
            profile, self.embeddings[expected_indices].mean(axis=0)
        )

    def test_rejects_empty_categories(self):
        with self.assertRaisesRegex(ValueError, "Select at least one category"):
            build_cold_start_profile(self.df, self.embeddings, [])

    def test_rejects_unknown_category(self):
        with self.assertRaisesRegex(ValueError, "No papers found"):
            build_cold_start_profile(
                self.df, self.embeddings, ["not-a-category"]
            )

    def test_rejects_mismatched_embedding_count(self):
        with self.assertRaisesRegex(ValueError, "same number of papers"):
            build_cold_start_profile(
                self.df, self.embeddings[:-1], ["cs.CV"]
            )

    def test_removes_duplicate_category_selections(self):
        _, indices = build_cold_start_profile(
            self.df, self.embeddings, ["cs.CL", "cs.CL"]
        )

        self.assertEqual(len(indices), 3)


if __name__ == "__main__":
    unittest.main()
