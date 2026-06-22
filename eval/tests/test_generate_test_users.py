import sys
import unittest
from pathlib import Path

import pandas as pd


EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from generate_test_users import (  # noqa: E402
    USER_TYPES,
    build_category_index,
    generate_days_ago,
    generate_synthetic_test_users,
    sample_papers_for_archetype,
    validate_users,
)


class GenerateTestUsersTests(unittest.TestCase):
    def setUp(self):
        categories = [
            "cs.CV",
            "cs.LG",
            "cs.RO",
            "cs.CL",
            "cs.IR",
            "cs.AI",
            "stat.ML",
            "math.OC",
        ]
        rows = []
        for category in categories:
            for index in range(80):
                rows.append(
                    {
                        "paper_id": f"{category}-{index}",
                        "primary_category": category,
                        "recency_score": 0.5,
                    }
                )
        self.df = pd.DataFrame(rows)

    def test_generation_is_reproducible_and_valid(self):
        first = generate_synthetic_test_users(self.df, n_users=50, seed=42)
        second = generate_synthetic_test_users(self.df, n_users=50, seed=42)

        self.assertEqual(first, second)
        validate_users(first, self.df)

    def test_user_history_size_matches_configured_user_type(self):
        users = generate_synthetic_test_users(self.df, n_users=100, seed=11)

        for user in users:
            lower, upper = USER_TYPES[user["user_type"]]["n_papers_range"]
            total_reading_list = len(user["train_history"]) + 1
            self.assertGreaterEqual(total_reading_list, lower)
            self.assertLessEqual(total_reading_list, upper)

    def test_history_is_unique_oldest_first_and_target_is_held_out(self):
        users = generate_synthetic_test_users(self.df, n_users=25, seed=9)

        for user in users:
            history_ids = [paper for paper, _ in user["train_history"]]
            days = [days_ago for _, days_ago in user["train_history"]]
            self.assertEqual(len(history_ids), len(set(history_ids)))
            self.assertEqual(days, sorted(days, reverse=True))
            self.assertNotIn(user["target_paper"], history_ids)
            self.assertNotIn(user["target_paper"], user["negative_papers"])

    def test_archetype_sampling_prefers_primary_and_secondary_pools(self):
        import random

        index = build_category_index(self.df)
        all_ids = self.df["paper_id"].tolist()
        archetype = {"primary": ["cs.CV"], "secondary": ["cs.LG"]}
        papers = sample_papers_for_archetype(
            all_ids, index, archetype, n_papers=20, rng=random.Random(2)
        )
        category_by_id = self.df.set_index("paper_id")[
            "primary_category"
        ].to_dict()
        categories = [category_by_id[paper] for paper in papers]

        self.assertGreaterEqual(categories.count("cs.CV"), 10)
        self.assertGreaterEqual(categories.count("cs.LG"), 2)
        self.assertEqual(len(papers), 20)
        self.assertEqual(len(set(papers)), 20)

    def test_days_ago_are_bounded_and_end_at_zero(self):
        import numpy as np

        days = generate_days_ago(30, np.random.default_rng(42))

        self.assertEqual(days[-1], 0)
        self.assertEqual(days, sorted(days, reverse=True))
        self.assertTrue(all(0 <= day <= 180 for day in days))


if __name__ == "__main__":
    unittest.main()
