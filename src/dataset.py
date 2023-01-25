import logging
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from math import fsum
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(init=False, repr=False)
class Dataset:
    """A general purpose dataset implementation that doesn't (more or less) rely on Pandas"""

    @dataclass(init=False)
    class Pairs:
        """A tuple of items having different classes"""

        pairs_list: list[tuple[int, int]]

        def __init__(self, dataset: npt.NDArray) -> None:
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

    class_probabilities: dict[str, float]
    costs: dict[str, float]
    features: list[str]
    _pairs: Pairs
    _table: npt.NDArray

    def __init__(self, dataset_path: Path) -> None:
        logger.info("Initializing dataset")
        dataset_df: pd.DataFrame = pd.read_csv(dataset_path)

        if "Probability" not in dataset_df.columns:
            number_of_rows = len(dataset_df.index)
            dataset_df["Probability"] = [1 / number_of_rows] * number_of_rows

        dataset_df.insert(loc=0, column="Index", value=range(dataset_df.shape[0]))

        dataset_np = dataset_df.to_numpy()

        self.features = dataset_df.columns.values[:-2].tolist()

        counter = Counter(dataset_np[:, -1, None].flatten().tolist())
        self.class_probabilities = {
            key: (value / sum(counter.values())) for key, value in counter.items()
        }

        self._pairs = self.Pairs(dataset_np)
        self._table = np.append([dataset_df.columns.values], dataset_np, axis=0)

        self.costs = {}

        # FIXME: Cannot use variance, it sums up to 1.
        #       To avoid the problem, we multiply by 10 the variance.
        #       Other cost metrics should be considered!
        for idx, column_name in enumerate(self.features):
            self.costs[column_name] = round(dataset_np[:, idx, None].var(), 2) * 10

        self.costs = dict(sorted(self.costs.items(), key=lambda item: item[1]))

        del counter
        del dataset_df, dataset_np

    def __getitem__(self, pos) -> npt.NDArray:
        """[] operator overload"""
        return self._table.__getitem__(pos)

    def __repr__(self) -> str:
        return self._table.__repr__()

    def get_class_of(self, idx: int) -> str:
        """Returns the class of the observation at index idx"""
        return self._table[idx + 1][-2]

    def copy(self) -> Self:
        """Returns a deep copy of the dataset"""
        return deepcopy(self)

    def data(self, *, complete=False) -> npt.NDArray:
        """Removes useless infos from dataset and returns it

        Args:
            complete (bool, optional): States if the output should contain 'Class' feature. Defaults to False.

        Returns:
            npt.NDArray: The content of the dataset.
        """
        if complete:
            return self._table[1:]

        return self._table[1:, :-2]

    def from_features_subset(self, features: list[str]) -> Self:
        """Returns a copy of the dataset, containing only the given features

        Args:
            features (npt.NDArray): The features to be returned

        Returns:
            Self: An instance of the dataset containing only the given features
        """
        if not features:
            return self.copy()

        logger.info(f"Generating dataset from subset {features} of features")
        dataset_copy = self.copy()
        remaining_features = set(self.features) - set(features)

        for feature in remaining_features:
            dataset_copy.drop_column(feature)

        return dataset_copy

    def drop_column(self, feature: str) -> None:
        """Removes the column with given label

        Args:
            feature (str): Label of the column to remove
        """
        logger.info("Dropping column %s", feature)
        feature_index = self.features.index(feature)
        self._table = np.delete(self.data(complete=True), feature_index, axis=1)
        self.features.remove(feature)

    def drop_row(self, index: int) -> None:
        """Removes the row at given index

        Args:
            index (int): Index of the row to remove
        """
        logger.info("Dropping row %i", index)
        self._table = np.delete(self.data(), index, axis=0)
        self._pairs.pairs_list = [
            pair for pair in self._pairs.pairs_list if index not in pair
        ]

    def difference(self, other: npt.NDArray, *, axis=0) -> npt.NDArray:
        """Computes the set difference between two datasets

        Args:
            other (npt.NDArray): The set to subtract from the dataset
            axis (int, optional): 0 for rows, 1 for columns. Defaults to 0.

        Returns:
            npt.NDArray: the set difference between two datasets
        """
        logger.debug("Computing datasets difference")
        return np.delete(self.data(), other, axis)

    def index_of_row(self, other: npt.NDArray) -> int | list[int]:
        """Returns the index of a given row

        Args:
            other (npt.NDArray): The row which we want to index

        Returns:
            int | list[int]: The row index
        """
        return np.where(np.any(self.data() == other))[0][0]

    def intersection(self, other: list[int]) -> Self:
        """Calculates the intersection between the dataset and the given subset of rows

        Args:
            other (npt.NDArray): The rows to intersect

        Returns:
            npt.NDArray: The resulting intersection
        """
        logger.info("Computing datasets intersection")
        dataset_copy = self.copy()

        data_as_set = set(self.indexes)
        difference = data_as_set.symmetric_difference(set(other))

        for row in difference:
            dataset_copy.drop_row(row)

        return dataset_copy

    def pairs_number_for(self, objects: list[int]) -> int:
        """Number of pairs containing objects

        Args:
            objects (list[int]): The objects to be checked

        Returns:
            int: Number of pairs
        """
        return len({pair for obj in objects for pair in self.pairs_list if obj in pair})

    @property
    def classes(self) -> list[str]:
        """Returns a list of all the possible class labels"""
        return list(self.class_probabilities.keys())

    @property
    def indexes(self) -> npt.NDArray:
        return self._table[:, None, 0].flatten()

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
    def total_cost(self) -> int:
        """Total costs of features

        Returns:
            int: Sum of all the costs
        """
        return int(fsum(self.costs.values()))

    @property
    def total_probability(self) -> float:
        """Total probability of the dataset objects

        Returns:
            float: Sum of all the probabilities
        """
        return fsum(probability for probability in self._table[1:, -1, None])
