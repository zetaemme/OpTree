from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import pandas as pd
from numpy import append, ndarray

Self = TypeVar('Self', bound='Dataset')


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
            item_classes: list[str] = dataset[:, -1, None].ravel().tolist()

            self.classes: list[str] = list(set(item_classes))

            if dataset.shape[0] == 1:
                self.pairs_list = []
                self.number = 0

                del item_classes

                return

            self.pairs_list: list[tuple[int, int]] = [
                (idx1, idx1 + idx2)
                for idx1, class1 in enumerate(item_classes)
                for idx2, class2 in enumerate(item_classes[idx1:])
                if class1 != class2
            ]

            del item_classes

            self.number: int = len(self.pairs_list)

    features: ndarray
    costs: dict[str, float]
    _pairs: Pairs
    _table: ndarray

    def __init__(self, dataset_path: Path) -> None:
        dataset_df: pd.DataFrame = pd.read_csv(dataset_path)
        dataset_np = dataset_df.to_numpy()

        self.features: ndarray = dataset_df.columns.values[:-1]
        self._pairs = self.Pairs(dataset_np)
        self._table: ndarray = append([dataset_df.columns.values], dataset_np, axis=0)

        self.costs = {}

        for idx, column_name in enumerate(self.features):
            self.costs[column_name] = round(dataset_np[:, idx, None].var())

        self.costs = {key: value for key, value in sorted(self.costs.items(), key=lambda item: item[1])}

        del dataset_df, dataset_np

    def __getitem__(self, idx: int) -> ndarray:
        """[] operator overload"""
        return self._table[idx + 1]

    def __repr__(self) -> str:
        return self._table.__repr__()

    def get_class_of(self, idx: int) -> str:
        """Returns the class of the observation at index idx"""
        return self._table[idx + 1][-1]

    def copy(self) -> Self:
        """Returns a deep copy of the dataset"""
        return deepcopy(self)

    def data(self, *, complete=False) -> ndarray:
        """
        Returns the dataset as a numpy.ndarray.
        If complete, the 'class column is returned as well.'
        """
        if complete:
            return self._table[1:]

        return self._table[1:, :-1]

    def multi_get(self, indexes: list[int], *, complete=False) -> ndarray:
        """Returns all the items specified in indexes"""
        return self.data(complete=complete)[[indexes]][0]

    @property
    def classes(self) -> list[str]:
        """Returns a list of all the possible class labels"""
        return self._pairs.classes

    @property
    def pairs_list(self) -> list[tuple]:
        """Returns a list of all the pairs of the dataset"""
        return self._pairs.pairs_list

    @property
    def pairs_number(self) -> int:
        """Return the number of pairs in the dataset"""
        return self._pairs.number

    @property
    def total_cost(self) -> int:
        """Returns the sum of the feature costs"""
        return sum(self.costs.values())
