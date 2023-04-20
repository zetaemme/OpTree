from pathlib import Path
from unittest import TestCase, main

from src.dataset import Dataset


class TestPairs(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = Dataset(Path("data/test.csv"))

    def test_fields(self) -> None:
        self.assertEqual(self.dataset.pairs_number, 8)
        self.assertListEqual(
            self.dataset.pairs_list,
            [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4)]
        )

    def test_pairs_number_for(self) -> None:
        self.assertEqual(
            self.dataset.pairs_number_for([0, 1, 3]),
            2
        )

        self.assertEqual(
            self.dataset.pairs_number_for([2, 4]),
            1
        )


if __name__ == '__main__':
    main()
