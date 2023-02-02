from numpy import array
from pathlib import Path
from unittest import TestCase, main

from src.dataset import Dataset


class TestDataset(TestCase):
    def setUp(self):
        self.dataset = Dataset(Path("tests/data/test.csv"))

    def test_fields(self):
        default_features = ["t1", "t2", "t3"]
        default_classes = {0: "A", 1: "A", 2: "B", 3: "C", 4: "C"}
        default_probabilities = [0.1, 0.2, 0.4, 0.25, 0.05]
        default_data = array([
            [0, 1, 1, 2],
            [1, 1, 2, 1],
            [2, 2, 2, 1],
            [3, 1, 2, 2],
            [4, 2, 2, 2]
        ])

        self.assertListEqual(self.dataset.features, default_features)
        self.assertDictEqual(self.dataset.classes, default_classes)
        self.assertListEqual(self.dataset._probabilities, default_probabilities)
        self.assertEqual(self.dataset.total_probability, sum(default_probabilities))
        self.assertEqual(self.dataset.data().tolist(), default_data.tolist())


if __name__ == '__main__':
    main()
