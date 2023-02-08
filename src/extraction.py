from src.dataset import Dataset
from src.separation import Separation


def cheapest_separation(dataset: Dataset, separation: Separation, obj_idx1: int, obj_idx2: int, ) -> str:
    for feature, _ in sorted(dataset.costs.items(), key=lambda x: x[1]):
        if tuple([obj_idx1, obj_idx2]) in separation.separated[feature]:
            return feature
