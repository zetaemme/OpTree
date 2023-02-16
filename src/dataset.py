import logging
import numbers
from copy import deepcopy
from dataclasses import dataclass, field
from functools import reduce
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

    @dataclass(init=False)
    class Separation:
        _all_features: list[str]

        S_label: dict[str, dict[Any, list[int]]] = field(default_factory=dict)
        S_star: dict[str, list[int]] = field(default_factory=dict)
        sigma: dict[str, list[int]] = field(default_factory=dict)

        kept: dict[str, list[tuple[int]]] = field(default_factory=dict)
        separated: dict[str, list[tuple[int]]] = field(default_factory=dict)

        def __init__(self, dataset: 'Dataset') -> None:
            logger.debug("Computing dataset separation")
            self.S_label = {}
            self.S_star = {}
            self.sigma = {}
            self.kept = {}
            self.separated = {}

            self._all_features = dataset.features

            for feature_idx, feature in enumerate(self._all_features):
                self.S_label[feature] = {
                    value: [observation[0] for observation in dataset.data() if observation[feature_idx + 1] == value]
                    for value in set(dataset.data()[:, feature_idx + 1])
                }

                feature_pairs = {
                    label: dataset.pairs_number_for(objects) for label, objects in self.S_label[feature].items()
                }
                max_pairs = max(feature_pairs, key=feature_pairs.get)  # type: ignore
                self.S_star[feature] = self.S_label[feature][max_pairs]

                self.sigma[feature] = [row[0] for row in dataset.difference(self.S_star[feature])]

                # FIXME: Find a better way to do this (itertools)
                self.kept[feature] = list(
                    {
                        pair
                        for pair in dataset.pairs_list
                        if pair[0] in self.sigma[feature] and pair[1] in self.sigma[feature]
                    }
                )

                # FIXME: Find a better way to do this (itertools)
                self.separated[feature] = list(
                    {
                        pair
                        for pair in dataset.pairs_list
                        if pair[0] in self.sigma[feature]
                           and pair[1] in self.S_star[feature]
                           or pair[1] in self.sigma[feature]
                           and pair[0] in self.S_star[feature]
                    }
                )

        def __getitem__(self, key: str) -> dict[Any, list[int]]:
            return self.S_label[key]

        def for_features_subset(self, features: list[str]) -> Self:
            separation_copy = deepcopy(self)

            to_delete = set(self._all_features).difference(features)
            for feature in to_delete:
                del separation_copy.S_label[feature]
                del separation_copy.S_star[feature]
                del separation_copy.sigma[feature]
                del separation_copy.kept[feature]
                del separation_copy.separated[feature]

            return separation_copy

        @property
        def S_star_intersection(self) -> list[int]:
            """Returns the intersection on the tests of S^*_t"""
            return reduce(np.intersect1d, self.S_star.values())  # type: ignore

    # class_probabilities: dict[str, float]
    costs: dict[str, float]
    features: list[str]
    _classes: dict[int, str]
    _data: np.ndarray
    _header: list[str]
    _pairs: Pairs
    _probabilities: list[float]
    _separation: Separation

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

        self._pairs = self.Pairs(dataset_np)
        self._data = dataset_np[:, :-2]

        self.costs = {}

        # FIXME: Cannot use variance, it sums up to 1.
        #       To avoid the problem, we multiply by 10 the variance.
        #       Other cost metrics should be considered!
        for idx, column_name in enumerate(self.features):
            if isinstance(dataset_np[:, idx + 1][0], numbers.Number):
                # self.costs[column_name] = round(dataset_np[:, idx + 1].var(), 2) * 10
                self.costs[column_name] = 1
            else:
                self.costs[column_name] = 1

        self.costs = {"t1": 5, "t2": 0.1, "t3": 1}

        del dataset_df, dataset_np

        self._separation = self.Separation(self)

    def __getitem__(self, pos) -> np.ndarray:
        """[] operator overload"""
        return self._data.__getitem__(pos)

    def __len__(self) -> int:
        return self._data.__len__()

    def __repr__(self) -> str:
        full_data = np.vstack((np.array(self._header[:-2]), self._data))
        return full_data.__repr__()

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
        del self.S_label[feature]
        del self.S_star[feature]
        del self.sigma[feature]
        del self.separated[feature]
        del self.kept[feature]

    def drop_row(self, index: int) -> None:
        """Removes the row at given index

        Args:
            index (int): Index of the row to remove
        """
        logger.debug("Dropping row %i", index)
        drop_index = np.where(self.indexes == index)
        self._data = np.delete(self._data, drop_index[0][0], axis=0)
        self._pairs.pairs_list = [pair for pair in self._pairs.pairs_list if index not in pair]
        del self._classes[index]
        del self._probabilities[drop_index[0][0]]

        for feature in self.features:
            for label in self.S_label[feature]:
                try:
                    self.S_label[feature][label].remove(index)
                except ValueError:
                    pass

            try:
                self.S_star[feature].remove(index)
            except ValueError:
                pass

            try:
                self.sigma[feature].remove(index)
            except ValueError:
                pass

            self.separated[feature] = [
                pair
                for pair in self.separated[feature]
                if index not in pair
            ]

            self.kept[feature] = [
                pair
                for pair in self.kept[feature]
                if index not in pair
            ]

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

    def print_separation(self) -> None:
        print()
        print(f"S_label: {self.S_label}")
        print(f"S_star: {self.S_star}")
        print(f"sigma: {self.sigma}")
        print(f"separated: {self.separated}")
        print(f"kept: {self.kept}")
        print()

    @property
    def classes(self) -> dict[int, str]:
        """Returns a dict of all the possible classes with relative object index"""
        return self._classes

    @property
    def indexes(self) -> np.ndarray:
        return self._data[:, 0]  # type: ignore

    @property
    def kept(self) -> dict[str, list[tuple[int]]]:
        return self._separation.kept

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
    def S_label(self) -> dict[str, dict[Any, list[int]]]:
        return self._separation.S_label

    @property
    def S_star(self) -> dict[str, list[int]]:
        return self._separation.S_star

    @property
    def separated(self) -> dict[str, list[tuple[int]]]:
        return self._separation.separated

    def separation_for_features_subset(self, features: list[str]) -> Separation:
        return self._separation.for_features_subset(features)

    @property
    def sigma(self) -> dict[str, list[int]]:
        return self._separation.sigma

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
