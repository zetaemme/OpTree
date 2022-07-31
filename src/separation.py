from dataclasses import dataclass, field
from typing import Any

from src.dataset import Dataset


@dataclass(init=False)
class Separation:
    features_separation: dict[str: dict[Any, list[int]]] = field(default_factory=dict)

    def __init__(self, dataset: Dataset) -> None:
        self.features_separation = {
            feature: {
                value: [
                    obj_idx
                    for obj_idx, obj_value in enumerate(dataset.data)
                    if obj_value[feature_idx] == value
                ]
                for value in {value[0] for value in dataset.data[:, feature_idx, None]}
            } for feature_idx, feature in enumerate(dataset.features)
        }

    def __getitem__(self, key: str) -> dict[Any, list[int]]:
        return self.features_separation[key]

    def check(self, feature: str, obj_idx1: int, obj_idx2: int) -> bool:
        key_list = list(self.features_separation[feature])

        for idx, key1 in enumerate(key_list):
            for key2 in key_list[idx:]:
                if (
                        obj_idx1 in self.features_separation[feature][key1] and
                        obj_idx2 in self.features_separation[feature][key2]
                ) or (
                        obj_idx1 in self.features_separation[feature][key2] and
                        obj_idx2 in self.features_separation[feature][key1]
                ):
                    return True

                continue
