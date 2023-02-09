import logging
# from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
from math import fsum
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(init=False, repr=False)
class Dataset:
    """A general purpose dataset implementation that doesn't (more or less) rely on Pandas"""

    @dataclass(init=False)
    class Pairs:
        """A tuple of items having different classes"""

        pairs_list: list[tuple[int, int]]

        def __init__(self, dataset: np.ndarray) -> None:
            logger.info("Computing dataset pairs")
            item_classes: list[str] = dataset[:, -2, None].ravel().tolist()

            if dataset.shape[0] == 1:
                self.pairs_list = []

                del item_classes
                return

            self.pairs_list: list[tuple[int, int]] = [
                (idx1, idx1 + idx2)
                for idx1, class1 in enumerate(item_classes)
                for idx2, class2 in enumerate(item_classes[idx1:])
                if class1 != class2
            ]

            del item_classes

        @property
        def number(self) -> int:
            """Number of paris

            Returns:
                int: The total number of pairs for the dataset
            """
            return len(self.pairs_list)

    # class_probabilities: dict[str, float]
    costs: dict[str, float]
    features: list[str]
    _classes: dict[int, str]
    _data: np.ndarray
    _header: list[str]
    _pairs: Pairs
    _probabilities: list[float]

    def __init__(self, dataset_path: Path) -> None:
        logger.info("Initializing dataset")
        dataset_df: pd.DataFrame = pd.read_csv(dataset_path)

        # If no probability is given, assume uniform
        if "Probability" not in dataset_df.columns:
            number_of_rows = len(dataset_df.index)
            dataset_df["Probability"] = [1 / number_of_rows] * number_of_rows

        self._classes = dataset_df["Class"].to_dict()
        self._probabilities = dataset_df["Probability"].to_list()

        # Add "Index" column
        dataset_df.insert(loc=0, column="Index", value=range(dataset_df.shape[0]))  # type: ignore

        # Extract dataset's features
        self._header = dataset_df.columns.to_list()
        self.features = self._header[1:-2]

        dataset_np = dataset_df.to_numpy()

        # Calculate probability of each class
        # counter = Counter(dataset_np[:, -1, None].flatten().tolist())
        # self.class_probabilities = {key: (value / sum(counter.values())) for key, value in counter.items()}

        self._pairs = self.Pairs(dataset_np)
        self._data = dataset_np[:, :-2]

        self.costs = {}

        # FIXME: Cannot use variance, it sums up to 1.
        #       To avoid the problem, we multiply by 10 the variance.
        #       Other cost metrics should be considered!
        for idx, column_name in enumerate(self.features):
            self.costs[column_name] = round(dataset_np[:, idx + 1].var(), 2) * 10

        # del counter
        del dataset_df, dataset_np

    def __getitem__(self, pos) -> np.ndarray:
        """[] operator overload"""
        return self._data.__getitem__(pos)

    def __len__(self) -> int:
        return self._data.__len__()

    def __repr__(self) -> str:
        return self._data.__repr__()

    def copy(self) -> Self:
        """Returns a deep copy of the dataset"""
        return deepcopy(self)

    def data(self) -> np.ndarray:
        """Removes useless infos from dataset and returns it

        Returns:
            npt.NDArray: The content of the dataset.
        """
        return self._data

    def drop_feature(self, feature: str) -> None:
        feature_index = self.features.index(feature)

        self._data = np.delete(self._data, feature_index + 1, 1)
        self.features.remove(feature)
        self._header.remove(feature)
        del self.costs[feature]

    def drop_row(self, index: int) -> None:
        """Removes the row at given index

        Args:
            index (int): Index of the row to remove
        """
        logger.debug("Dropping row %i", index)
        drop_index = np.where(self.indexes == index)
        self._data = np.delete(self._data, drop_index[0][0], axis=0)
        self._pairs.pairs_list = [pair for pair in self._pairs.pairs_list if index not in pair]

    def difference(self, other: list[int], *, axis=0) -> np.ndarray:
        """Computes the set difference between two datasets

        Args:
            other (npt.NDArray): The set to subtract from the dataset
            axis (int, optional): 0 for rows, 1 for columns. Defaults to 0.

        Returns:
            npt.NDArray: the set difference between two datasets
        """
        logger.debug("Computing datasets difference")
        drop_indexes = np.array([
            arr
            for arr in chain(
                np.where(self.indexes == index)
                for index in other
            )
        ]).flatten()
        return np.delete(self.data(), drop_indexes, axis)

    def intersection(self, other: list[int]) -> Self:
        """Calculates the intersection between the dataset and the given subset of rows

        Args:
            other (list[int]): Indexes of the rows to intersect

        Returns:
            Dataset: The resulting intersection
        """
        logger.debug("Computing datasets intersection")
        dataset_copy = self.copy()

        data_as_set = set(self.indexes)

        difference = data_as_set - set(other)

        for row in difference:
            dataset_copy.drop_row(row)

        return dataset_copy

    def labels_for(self, feature: str) -> set[Any]:
        feature_index = self.features.index(feature)
        return set(self._data[:, feature_index + 1])

    def pairs_number_for(self, objects: list[int]) -> int:
        """Number of pairs containing objects

        Args:
            objects (list[int]): The objects to be checked

        Returns:
            int: Number of pairs
        """
        return len({pair for obj in objects for pair in self.pairs_list if obj in pair})

    @property
    def classes(self) -> dict[int, str]:
        """Returns a dict of all the possible classes with relative object index"""
        return self._classes

    @property
    def indexes(self) -> np.ndarray:
        return self._data[:, 0]  # type: ignore

    @property
    def mean(self) -> float:
        return self.data().mean()

    @property
    def pairs_list(self) -> list[tuple]:
        """Returns a list of all the pairs of the dataset"""
        return self._pairs.pairs_list

    @property
    def pairs_number(self) -> int:
        """Return the number of pairs in the dataset"""
        return self._pairs.number

    @property
    def std(self) -> float:
        return self.data().std()

    @property
    def total_cost(self) -> float:
        """Total costs of features

        Returns:
            int: Sum of all the costs
        """
        return fsum(self.costs.values())

    @property
    def total_probability(self) -> float:
        """Total probability of the dataset objects

        Returns:
            float: Sum of all the probabilities
        """
        return fsum(probability for probability in self._probabilities)
