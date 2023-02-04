from pathlib import Path
from unittest import TestCase, main

import numpy as np

from src.dataset import Dataset


class TestDataset(TestCase):
    def setUp(self):
        self.dataset = Dataset(Path("data/test.csv"))

    def test_fields(self) -> None:
        default_features = ["t1", "t2", "t3"]
        default_classes = {0: "A", 1: "A", 2: "B", 3: "C", 4: "C"}
        default_probabilities = [0.1, 0.2, 0.4, 0.25, 0.05]
        default_data = [
            [0, 1, 1, 2],
            [1, 1, 2, 1],
            [2, 2, 2, 1],
            [3, 1, 2, 2],
            [4, 2, 2, 2]
        ]

        self.assertListEqual(self.dataset.features, default_features)
        self.assertDictEqual(self.dataset.classes, default_classes)
        self.assertListEqual(self.dataset._probabilities, default_probabilities)
        self.assertEqual(self.dataset.total_probability, sum(default_probabilities))
        self.assertEqual(self.dataset.data().tolist(), default_data)

    def test_drop_row(self) -> None:
        dataset_copy = self.dataset.copy()
        dataset_copy.drop_row(0)
        self.assertEqual(
            dataset_copy.data().tolist(),
            [
                [1, 1, 2, 1],
                [2, 2, 2, 1],
                [3, 1, 2, 2],
                [4, 2, 2, 2]
            ],
            "Error dropping row 0"
        )

        dataset_copy = self.dataset.copy()
        dataset_copy.drop_row(1)
        self.assertEqual(
            dataset_copy.data().tolist(),
            [
                [0, 1, 1, 2],
                [2, 2, 2, 1],
                [3, 1, 2, 2],
                [4, 2, 2, 2]
            ],
            "Error dropping row 1"
        )

        dataset_copy = self.dataset.copy()
        dataset_copy.drop_row(2)
        self.assertEqual(
            dataset_copy.data().tolist(),
            [
                [0, 1, 1, 2],
                [1, 1, 2, 1],
                [3, 1, 2, 2],
                [4, 2, 2, 2]
            ],
            "Error dropping row 2"
        )

        dataset_copy = self.dataset.copy()
        dataset_copy.drop_row(3)
        self.assertEqual(
            dataset_copy.data().tolist(),
            [
                [0, 1, 1, 2],
                [1, 1, 2, 1],
                [2, 2, 2, 1],
                [4, 2, 2, 2]
            ],
            "Error dropping row 3"
        )

        dataset_copy = self.dataset.copy()
        dataset_copy.drop_row(4)
        self.assertEqual(
            dataset_copy.data().tolist(),
            [
                [0, 1, 1, 2],
                [1, 1, 2, 1],
                [2, 2, 2, 1],
                [3, 1, 2, 2]
            ],
            "Error dropping row 4"
        )

    def test_difference(self) -> None:
        other = Dataset(Path("data/test.csv"))
        other.drop_row(2)
        other.drop_row(3)

        difference = self.dataset.difference(other.indexes.tolist()).tolist()  # type: ignore

        self.assertEqual(
            difference,
            [
                [2, 2, 2, 1],
                [3, 1, 2, 2]
            ],
            "Error computing difference without rows 2 and 3"
        )


if __name__ == '__main__':
    main()
