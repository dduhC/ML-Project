import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from user_profile_generation import (
    build_paper_id_to_index,
    compute_user_profile,
)


class UserProfileGenerationTests(unittest.TestCase):
    def setUp(self):
        self.paper_ids = ["paper-z", "paper-a", "paper-m"]
        self.embeddings = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        self.user = {
            "user_id": "user-test",
            "train_history": [["paper-a", 0], ["paper-z", 30]],
        }

    def test_mapping_uses_embedding_row_order_not_sorted_ids(self):
        mapping = build_paper_id_to_index(self.paper_ids)

        self.assertEqual(mapping, {"paper-z": 0, "paper-a": 1, "paper-m": 2})

    def test_dense_profile_uses_correct_paper_rows(self):
        profile, metadata = compute_user_profile(
            self.user,
            self.embeddings,
            build_paper_id_to_index(self.paper_ids),
            half_life=30,
        )

        np.testing.assert_allclose(profile, [1 / 3, 2 / 3])
        self.assertEqual(metadata["missing_papers"], [])

    def test_sparse_profile_matches_dense_profile(self):
        mapping = build_paper_id_to_index(self.paper_ids)
        dense, _ = compute_user_profile(
            self.user, self.embeddings, mapping, half_life=30
        )
        sparse, _ = compute_user_profile(
            self.user,
            scipy.sparse.csr_matrix(self.embeddings),
            mapping,
            half_life=30,
        )

        np.testing.assert_allclose(sparse, dense)

    def test_rejects_duplicate_dataset_paper_ids(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            build_paper_id_to_index(["paper-a", "paper-a"])


if __name__ == "__main__":
    unittest.main()
