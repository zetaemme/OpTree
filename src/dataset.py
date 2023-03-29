import logging
import numbers
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property, reduce
from itertools import chain, combinations
from math import ceil, fsum
from pathlib import Path
from pickle import HIGHEST_PROTOCOL, dump
from typing import Any, Literal, Self

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(init=False, repr=False)
class Dataset:
    """A dataset implementation that doesn't (more or less) rely on Pandas"""

    @dataclass(init=False)
    class Pairs:
        """A tuple of items having different classes"""

        pairs_list: list[tuple[int, int]]

        def __init__(self, dataset: np.ndarray, name: str = None) -> None:
            logger.debug("Computing dataset pairs")
            if dataset.shape[0] == 1:
                self.pairs_list = []
                return

            self.pairs_list = list(
                filter(
                    lambda pair: dataset[pair[0], -2] != dataset[pair[1], -2],
                    combinations(dataset[:, 0], 2)
                )
            )

            # Saves the pairs, so we don't need to recompute them in future executions
            if name is not None and not Path(f"./data/pairs/{name}_pairs.pkl").is_file():
                with open(f"./data/pairs/{name}_pairs.pkl", "w") as f:
                    dump({"pairs": self.pairs_list}, f, HIGHEST_PROTOCOL)

        @classmethod
        def from_precomputed(cls, pairs: list[tuple[int]]) -> Self:
            """Constructs a Pairs object from the pre-computed pickled file

            Args:
                pairs: The pre-computed Pairs object

            Returns:
                An instance of Pairs from the pre-computed object
            """
            # NOTE: This is a workaround to construct a Pairs object without an explicit call to the constructor.
            #       Needed since we allow a pre-computation of the pairs for the dataset, giving it as an input to
            #       the procedure.
            pairs_obj = cls(np.ndarray((1, 1)))
            pairs_obj.pairs_list = pairs
            return pairs_obj

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

        def __init__(self, dataset: 'Dataset', name: str = None) -> None:
            if not dataset._path:
                return

            logger.debug("Computing dataset separation")
            self.S_label = {}
            self.S_star = {}
            self.sigma = {}
            self.kept = {}
            self.separated = {}

            self._all_features = dataset.features

            for feature_idx, feature in enumerate(self._all_features):
                self.S_label[feature] = {
                    value: [item[0] for item in dataset.data() if item[feature_idx + 1] == value]
                    for value in set(dataset.data()[:, feature_idx + 1])
                }

                feature_pairs = {
                    label: dataset.pairs_number_for(objects) for label, objects in self.S_label[feature].items()
                }
                # type: ignore
                max_pairs = max(feature_pairs, key=feature_pairs.get)
                self.S_star[feature] = self.S_label[feature][max_pairs]

                self.sigma[feature] = [row[0]
                                       for row in dataset.difference(self.S_star[feature])]

                self.kept[feature] = list(
                    filter(
                        lambda pair: all(
                            idx in self.sigma[feature] for idx in pair),
                        dataset.pairs_list
                    )
                )

                self.separated[feature] = list(
                    filter(
                        lambda pair: pair[0] in self.sigma[feature] and pair[1] in self.S_star[feature] or
                        pair[1] in self.sigma[feature] and pair[0] in self.S_star[feature],
                        dataset.pairs_list
                    )
                )

            # Saves the pairs, so we don't need to recompute them in future executions
            if name is not None and not Path(f"./data/separation/{name}_separation.pkl").is_file():
                with open(f"./data/separation/{name}_separation.pkl", "w") as f:
                    dump(
                        {
                            "S_label": self.S_label,
                            "S_star": self.S_star,
                            "sigma": self.sigma,
                            "separated": self.separated,
                            "kept": self.kept
                        },
                        f,
                        HIGHEST_PROTOCOL
                    )

        def __getitem__(self, key: str) -> dict[Any, list[int]]:
            return self.S_label[key]

        @classmethod
        def from_precomputed(
                cls,
                separation: dict[
                    Literal["S_label", "S_star", "sigma", "separated", "kept"],
                    dict[str, list[Any | tuple]]
                ],
                features: list[str]
        ) -> Self:
            """Constructs a Separation object from the pre-computed pickled file

            Args:
                separation: The pre-computed Separation object
                features(list[str]): The features of the dataset

            Returns:
                An instance of Separation from the pre-computed object
            """
            separation_obj = cls(Dataset(Path(""), None, None))

            separation_obj.S_label = separation["S_label"]
            separation_obj.S_star = separation["S_star"]
            separation_obj.sigma = separation["sigma"]
            separation_obj.separated = separation["separated"]
            separation_obj.kept = separation["kept"]
            separation_obj._all_features = features

            return separation_obj

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

        @cached_property
        def S_star_intersection(self) -> list[int]:
            """Returns the intersection on the tests of S^*_t"""
            return reduce(np.intersect1d, self.S_star.values())  # type: ignore

    costs: dict[str, float]
    features: list[str]
    _classes: dict[int, str]
    _data: np.ndarray
    _header: list[str]
    _pairs: Pairs
    _path: str
    _probabilities: list[float]
    _separation: Separation

    def __init__(
            self,
            dataset_path: Path,
            pairs: dict[Literal["pairs"], list[list[int]]] | None = None,
            separation: dict[
                Literal["S_label", "S_star", "sigma", "separated", "kept"],
                dict[str, list[Any | tuple]]
            ] | None = None
    ) -> None:
        self._path = dataset_path.name
        if dataset_path.name == "":
            return

        logger.info("Initializing dataset")
        dataset_df: pd.DataFrame = pd.read_csv(dataset_path)

        # If no probability is given, assume uniform
        if "Probability" not in dataset_df.columns:
            number_of_rows = len(dataset_df.index)
            dataset_df["Probability"] = [1 / number_of_rows] * number_of_rows

        self._classes = dataset_df["Class"].to_dict()
        self._probabilities = dataset_df["Probability"].to_list()

        # Add "Index" column
        dataset_df.insert(loc=0, column="Index", value=range(
            dataset_df.shape[0]))  # type: ignore

        # Extract dataset's features and "header"
        self._header = dataset_df.columns.to_list()
        self.features = self._header[1:-2]

        dataset_np = dataset_df.to_numpy()

        if pairs is None:
            logger.info("No pairs file found")
            self._pairs = self.Pairs(dataset_np, dataset_path.stem)
        else:
            logger.info("Using pairs from Pickle file")
            self._pairs = self.Pairs.from_precomputed(pairs)  # type: ignore

        self._data = dataset_np[:, :-2]

        self.costs = {}

        # NOTE: Cannot use variance, it sums up to 1.
        #        To avoid the problem, we multiply by 10 the variance.
        #        Other cost metrics should be considered!
        for idx, column_name in enumerate(self.features):
            if isinstance(dataset_np[:, idx + 1][0], numbers.Number):
                self.costs[column_name] = len(np.unique(self._data[:, idx + 1])) + (
                    ceil(self._data[:, idx + 1].var() * 10)
                )
                # self.costs[column_name] = 1
            else:
                self.costs[column_name] = len(
                    np.unique(self._data[:, idx + 1]))
                # self.costs[column_name] = 1

        del dataset_df, dataset_np

        if separation is None:
            logger.info("No separation file found")
            self._separation = self.Separation(self, dataset_path.stem)
        else:
            logger.info("Using separation from Pickle file")
            self._separation = self.Separation.from_precomputed(
                separation, self.features)

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
            ndarray: The content of the dataset.
        """
        return self._data

    def drop_feature(self, feature: str) -> None:
        if len(self.features) == 1 and self.features[0] == feature:
            self._data = np.ndarray([])
            self._header = []
            # self._classes = {}
            self._probabilities = []

            self._pairs.pairs_list = []

            return

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
            ndarray: the set difference between two datasets
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

    def labels_for(self, feature: str) -> np.ndarray:
        return np.unique(self._data[:, self.features.index(feature) + 1])

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

    def separation_for_features_subset(self, features: list[str]) -> Separation:
        return self._separation.for_features_subset(features)

    def without_feature(self, feature: str) -> Self:
        dataset_copy = self.copy()
        dataset_copy.drop_feature(feature)
        return dataset_copy

    @property
    def classes(self) -> dict[int, str]:
        """Returns a dict of all the possible classes with relative object index"""
        return self._classes

    @property
    def indexes(self) -> np.ndarray:
        return self._data[:, 0]  # type: ignore

    @cached_property
    def kept(self) -> dict[str, list[tuple[int]]]:
        return self._separation.kept

    @property
    def pairs_list(self) -> list[tuple]:
        """Returns a list of all the pairs of the dataset"""
        return self._pairs.pairs_list

    @property
    def pairs_number(self) -> int:
        """Return the number of pairs in the dataset"""
        return self._pairs.number

    @cached_property
    def S_label(self) -> dict[str, dict[Any, list[int]]]:
        return self._separation.S_label

    @cached_property
    def S_star(self) -> dict[str, list[int]]:
        return self._separation.S_star

    @cached_property
    def separated(self) -> dict[str, list[tuple[int]]]:
        return self._separation.separated

    @cached_property
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
