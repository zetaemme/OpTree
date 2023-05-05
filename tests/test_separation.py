from pathlib import Path
from unittest import TestCase, main

from src.dataset import Dataset


class TestSeparation(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = Dataset(Path("data/test.csv"))

    def test_fields(self) -> None:
        self.assertDictEqual(
            self.dataset.S_label,
            {
                "t1": {
                    "1": [0, 1, 3],
                    "2": [2, 4]
                },
                "t2": {
                    "1": [0],
                    "2": [1, 2, 3, 4]
                },
                "t3": {
                    "1": [1, 2],
                    "2": [0, 3, 4]
                },
            }
        )

        self.assertDictEqual(
            self.dataset.S_star,
            {
                "t1": [0, 1, 3],
                "t2": [1, 2, 3, 4],
                "t3": [0, 3, 4]
            }
        )

        self.assertDictEqual(
            self.dataset.sigma,
            {
                "t1": [2, 4],
                "t2": [0],
                "t3": [1, 2]
            }
        )


if __name__ == "__main__":
    main()
