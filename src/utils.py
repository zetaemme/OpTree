from numpy import ndarray, take

from src.dataset import Dataset
from src.separation import Separation


def submodular_function_1(dataset: Dataset, features: ndarray, separation: Separation) -> int:
    max_intersection: ndarray = separation.maximum_intersection(features)

    if len(max_intersection) == 1:
        return dataset.pairs_number

    intersection_pairs: Dataset.Pairs = Dataset.Pairs(take(dataset.data(complete=True), max_intersection, axis=0))

    return dataset.pairs_number - intersection_pairs.number
