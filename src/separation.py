import logging
from copy import deepcopy
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Self

from numpy import intersect1d

from src.dataset import Dataset

logger = logging.getLogger(__name__)


@dataclass(init=False)
class Separation:
    _all_features: list[str]

    S_label: dict[str, dict[Any, list[int]]] = field(default_factory=dict)
    S_star: dict[str, list[int]] = field(default_factory=dict)
    sigma: dict[str, list[int]] = field(default_factory=dict)

    kept: dict[str, list[tuple[int]]] = field(default_factory=dict)
    separated: dict[str, list[tuple[int]]] = field(default_factory=dict)

    def __init__(self, dataset: Dataset) -> None:
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

    def check(self, feature: str, obj_idx1: int, obj_idx2: int) -> bool:
        key_list = list(self.S_label[feature])

        for idx, key1 in enumerate(key_list):
            for key2 in key_list[idx:]:
                if (obj_idx1 in self.S_label[feature][key1] and obj_idx2 in self.S_label[feature][key2]) or (
                        obj_idx1 in self.S_label[feature][key2] and obj_idx2 in self.S_label[feature][key1]
                ):
                    return True

                continue

        return False

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
        return reduce(intersect1d, self.S_star.values())
