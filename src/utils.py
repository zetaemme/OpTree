from numpy import ndarray

from src.separation import Separation


def submodular_function_1(features: ndarray, pairs_number: int, separation: Separation) -> int:
    max_intersection = separation.maximum_intersection(features)

    if len(max_intersection) == 1:
        return pairs_number
