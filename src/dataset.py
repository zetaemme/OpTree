from dataclasses import dataclass
from numpy import ndarray, ndenumerate
import pandas as pd
from pathlib import Path


@dataclass(init=False)
class Dataset:
    """A general purpose dataset implementation that doesn't rely on Pandas"""

    @dataclass(init=False)
    class Pairs:
        """A tuple of items having different classes"""

        classes: list[str]
        number: int
        pairs_list: list[tuple[int, int]]

        def __init__(self, dataset: ndarray) -> None:
            item_classes = [(idx[0], value) for idx, value in ndenumerate(dataset) if idx[1] == dataset.shape[1] - 1]

            self.classes = list(set(class_name[1] for class_name in item_classes))

            self.pairs_list = [
                (idx1 + 1, idx2 + 1)
                for idx1, class1 in item_classes
                for idx2, class2 in item_classes[idx1:]
                if class1 != class2
            ]

            del item_classes

            self.number = len(self.pairs_list)

    _pairs: Pairs

    def __init__(self, dataset_path: Path) -> None:
        self._pairs: Dataset.Pairs = self.Pairs(pd.read_csv(dataset_path).to_numpy())

    @property
    def classes(self) -> list[str]:
        return self._pairs.classes

    @property
    def pairs_list(self) -> list[tuple]:
        return self._pairs.pairs_list

    @property
    def pairs_number(self) -> int:
        return self._pairs.number
