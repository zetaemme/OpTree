from typing import Callable

from numpy import delete, ndarray, union1d

from src.dataset import Dataset
from src.separation import Separation


def wolsey_greedy_heuristic(
        budget: int,
        dataset: Dataset,
        separation: Separation,
        submodular_function: Callable[[Dataset, ndarray, Separation], int]
) -> list[str]:
    # NOTE: This assumes that the whole set of features will be used.
    #       If not, just add 'features: ndarray' as argument
    features = dataset.features.copy()

    spent = 0.0
    k = 0

    auxiliary_array = ndarray([])

    while features or spent <= budget:
        k += 1

        features = delete(features, k)
        spent += dataset.costs[features[k]]

        auxiliary_array = union1d(auxiliary_array, features[k])

    if submodular_function(dataset, features[k], separation) \
            >= submodular_function(dataset, delete(auxiliary_array, k), separation):
        return features[k]

    return features[:k].tolist()
