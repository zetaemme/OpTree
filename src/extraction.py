from src.dataset import Dataset
from src.separation import Separation


def cheapest_separation(dataset: Dataset, obj_idx1: int, obj_idx2: int, separation: Separation) -> str:
    for feature in dataset.costs.keys():
        # FIXME: Paper says that (3, 4) should be separated for 't3', but they have the same value
        if separation.check(feature, obj_idx1, obj_idx2):
            return feature
