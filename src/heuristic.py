import logging

from src import memory
from src.dataset import Dataset
from src.maximization import submodular_maximization
from src.types import SubmodularFunction

logger = logging.getLogger(__name__)


@memory.cache
def wolsey_greedy_heuristic(
    budget: float,
    dataset: Dataset,
    submodular_function: SubmodularFunction,
) -> list[str]:
    """Implementation of the Wolsey greedy algorithm

    Args:
        budget (float): Budget threshold
        dataset (Dataset): Dataset
        submodular_function (SubmodularFunction): Submodular function to apply

    Returns:
        list[str]: A set of features
    """
    logger.info("Applying heuristic for budget: %f", budget)

    spent = 0.0
    auxiliary_array: list[str] = []
    k = -1

    heuristic_features = [
        feature for feature in dataset.features if dataset.costs[feature] <= budget
    ]

    if len(heuristic_features) > 0:
        while True:
            k += 1

            # Select t_k
            chosen_test = submodular_maximization(
                dataset,
                heuristic_features,
                auxiliary_array,
                submodular_function,
            )

            # Remove t_k from T and add it to A
            auxiliary_array.append(heuristic_features.pop(heuristic_features.index(chosen_test)))
            # Update spent
            spent += dataset.costs[chosen_test]

            if spent > budget or len(heuristic_features) == 0:
                break
    else:
        return []

    # Compute f({t_k})
    single_result = submodular_function(dataset, [auxiliary_array[k]])
    # Compute f(A \ {t_k})
    # NOTE: We can use [:-1] since t_K is the last element of the list
    difference_result = submodular_function(dataset, auxiliary_array[:-1])

    if single_result >= difference_result:
        # Return {t_k}
        return [auxiliary_array[k]]

    # Return {t_1, ..., t_(k - 1)}
    return dataset.features[:k]
