from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from numpy import append, ndarray


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

            self.pairs_list: list[tuple[int, int]] = [
                (idx1, idx1 + idx2)
                for idx1, class1 in enumerate(item_classes)
                for idx2, class2 in enumerate(item_classes[idx1:])
                if class1 != class2
            ]

            del item_classes

            self.number: int = len(self.pairs_list)

    columns: ndarray
    costs: dict[str, int]
    _pairs: Pairs
    _table: ndarray

    def __init__(self, dataset_path: Path) -> None:
        dataset_df: pd.DataFrame = pd.read_csv(dataset_path)
        dataset_np = dataset_df.to_numpy()

        self.columns: ndarray = dataset_df.columns.values[:-1]
        self._pairs: Dataset.Pairs = self.Pairs(dataset_np)
        self._table: ndarray = append([dataset_df.columns.values], dataset_np, axis=0)

        self.costs = {}

        for idx, column_name in enumerate(self.columns[:-1]):
            self.costs[column_name] = len(set(dataset_np[:, idx, None].flatten()))

        del dataset_df, dataset_np

    def __repr__(self) -> str:
        return self._table.__repr__()

    @property
    def classes(self) -> list[str]:
        return self._pairs.classes

    @property
    def pairs_list(self) -> list[tuple]:
        return self._pairs.pairs_list

    @property
    def pairs_number(self) -> int:
        return self._pairs.number
