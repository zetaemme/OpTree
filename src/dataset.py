from dataclasses import dataclass

import numpy as np
from numpy import ndarray, ndenumerate
import pandas as pd
from pathlib import Path


@dataclass(init=False, repr=False)
class Dataset:
    """A general purpose dataset implementation that doesn't rely on Pandas"""

    @dataclass(init=False)
    class Pairs:
        """A tuple of items having different classes"""

        classes: list[str]
        number: int
        pairs_list: list[tuple[int, int]]

        def __init__(self, dataset: ndarray) -> None:
            item_classes: list[tuple] = [(idx[0], value) for idx, value in ndenumerate(dataset) if idx[1] == dataset.shape[1] - 1]

            self.classes: list[str] = list(set(class_name[1] for class_name in item_classes))

            self.pairs_list: list[tuple[int, int]] = [
                (idx1 + 1, idx2 + 1)
                for idx1, class1 in item_classes
                for idx2, class2 in item_classes[idx1:]
                if class1 != class2
            ]

            del item_classes

            self.number: int = len(self.pairs_list)

    _columns: ndarray
    _pairs: Pairs
    _table: ndarray

    def __init__(self, dataset_path: Path) -> None:
        dataset_df: pd.DataFrame = pd.read_csv(dataset_path)

        self._columns: ndarray = dataset_df.columns.values
        self._pairs: Dataset.Pairs = self.Pairs(dataset_df.to_numpy())
        self._table: ndarray = np.append([dataset_df.columns.values], dataset_df.to_numpy(), axis=0)

        del dataset_df

    def __repr__(self) -> str:
        return self._table.__repr__()

    @property
    def classes(self) -> list[str]:
        return self._pairs.classes

    @property
    def columns(self) -> ndarray:
        return self._columns

    @property
    def pairs_list(self) -> list[tuple]:
        return self._pairs.pairs_list

    @property
    def pairs_number(self) -> int:
        return self._pairs.number
