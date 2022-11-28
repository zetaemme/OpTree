from typing import Callable

from src.dataset import Dataset
from src.maximization import submodular_maximization


def wolsey_greedy_heuristic(
        budget: int,
        dataset: Dataset,
        submodular_function: Callable[[Dataset, list[str]], int]
) -> list[str]:
    # NOTE: This assumes that the whole set of features will be used.
    #       If not, just add 'features: ndarray' as argument
    features = dataset.features

    spent = 0.0
    k = 0

    auxiliary_array: list[str] = []
    chosen_test = ""

    while features:
        if spent <= budget:
            k += 1

            chosen_test = submodular_maximization(
                dataset,
                auxiliary_array,
                submodular_function
            )

            features.remove(chosen_test)
            spent += dataset.costs[chosen_test]
            auxiliary_array.append(chosen_test)

    auxiliary_array.remove(chosen_test)
    if submodular_function(dataset, [chosen_test]) >= submodular_function(dataset, auxiliary_array):
        return [chosen_test]

    # FIXME: Must return all items before chosen_test!
    return features[:chosen_test]
