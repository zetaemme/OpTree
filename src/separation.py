from dataclasses import dataclass, field
from functools import reduce
from typing import Any

from numpy import intersect1d, ndarray

from src.dataset import Dataset


@dataclass(init=False)
class Separation:
    S_star: dict[str, list[int]] = field(default_factory=dict)
    """S^*_t with t test"""

    S_label: dict[str: dict[Any, list[int]]] = field(default_factory=dict)
    """S^i_t with i corresponding to the single labels in the feature column, t test"""

    def __init__(self, dataset: Dataset) -> None:
        self.S_label = {
            feature: {
                value: [
                    obj_idx
                    for obj_idx, obj_value in enumerate(dataset.data())
                    if obj_value[feature_idx] == value
                ]
                for value in {value[0] for value in dataset.data()[:, feature_idx, None]}
            } for feature_idx, feature in enumerate(dataset.features)
        }

        self.S_star = {}
        for test in dataset.features:
            feature_pairs = {
                label: Dataset.Pairs(dataset.multi_get(objects)).number
                for label, objects in self.S_label[test].items()
            }

            max_pairs = max(feature_pairs, key=feature_pairs.get)
            self.S_star[test] = self.S_label[test][max_pairs]

    def __getitem__(self, key: str) -> dict[Any, list[int]]:
        return self.S_label[key]

    def check(self, feature: str, obj_idx1: int, obj_idx2: int) -> bool:
        key_list = list(self.S_label[feature])

        for idx, key1 in enumerate(key_list):
            for key2 in key_list[idx:]:
                if (
                        obj_idx1 in self.S_label[feature][key1] and
                        obj_idx2 in self.S_label[feature][key2]
                ) or (
                        obj_idx1 in self.S_label[feature][key2] and
                        obj_idx2 in self.S_label[feature][key1]
                ):
                    return True

                continue

    @property
    def S_star_intersection(self) -> ndarray:
        """Returns the intersection on the tests of S^*_t"""
        return reduce(intersect1d, self.S_star.values())
