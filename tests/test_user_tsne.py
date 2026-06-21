import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from user_tsne import (
    create_user_profile_tsne,
    prepare_user_tsne_data,
    reduce_user_tsne,
)


class UserTsneTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.df = pd.DataFrame(
            {
                "primary_category": (
                    ["cs.CV"] * 15 + ["cs.LG"] * 15 + ["cs.CL"] * 15
                )
            }
        )
        self.embeddings = rng.normal(size=(45, 8))
        self.user_vector = rng.normal(size=8)

    def test_balanced_sampling_is_reproducible_and_excludes_highlights(self):
        kwargs = dict(
            df=self.df,
            embeddings=self.embeddings,
            user_vector=self.user_vector,
            recommendation_indices=[0, 15, 30],
            context_indices=[1, 16, 31],
            context_label="Reading history",
            samples_per_category=4,
            seed=42,
        )
        first = prepare_user_tsne_data(**kwargs)
        second = prepare_user_tsne_data(**kwargs)

        first_background = first[first["role"] == "Background"]
        second_background = second[second["role"] == "Background"]
        self.assertEqual(
            first_background["category"].value_counts().to_dict(),
            {"cs.CV": 4, "cs.LG": 4, "cs.CL": 4},
        )
        self.assertEqual(
            first_background["paper_index"].tolist(),
            second_background["paper_index"].tolist(),
        )
        self.assertTrue(
            set(first_background["paper_index"]).isdisjoint({0, 1, 15, 16, 30, 31})
        )

    def test_duplicate_context_and_recommendation_uses_recommendation_role(self):
        frame = prepare_user_tsne_data(
            self.df,
            self.embeddings,
            self.user_vector,
            recommendation_indices=[0],
            context_indices=[0, 1],
            context_label="Cold-start seed papers",
            samples_per_category=2,
        )

        paper_zero = frame[frame["paper_index"] == 0]
        self.assertEqual(len(paper_zero), 1)
        self.assertEqual(paper_zero.iloc[0]["role"], "Recommendation")

    def test_reduction_handles_small_sample_by_adjusting_perplexity(self):
        coordinates = reduce_user_tsne(
            np.eye(4, dtype=float),
            perplexity=30,
            seed=42,
        )

        self.assertEqual(coordinates.shape, (4, 2))
        self.assertTrue(np.isfinite(coordinates).all())

    def test_rejects_invalid_dimension_and_index(self):
        with self.assertRaisesRegex(ValueError, "dimension"):
            prepare_user_tsne_data(
                self.df,
                self.embeddings,
                np.ones(3),
                [0],
                [1],
                "Reading history",
            )

        with self.assertRaisesRegex(ValueError, "outside"):
            prepare_user_tsne_data(
                self.df,
                self.embeddings,
                self.user_vector,
                [100],
                [1],
                "Reading history",
            )

    def test_creates_and_saves_figure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "user_tsne.png"
            figure = create_user_profile_tsne(
                self.df,
                self.embeddings,
                self.user_vector,
                recommendation_indices=[0, 15, 30],
                context_indices=[1, 16, 31],
                context_label="Reading history",
                samples_per_category=4,
                perplexity=5,
                output_path=output,
            )

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)
            self.assertEqual(len(figure.axes), 1)


if __name__ == "__main__":
    unittest.main()
