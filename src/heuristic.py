import logging

from src.dataset import Dataset
from src.maximization import submodular_maximization
from src.separation import Separation
from src.types import SubmodularFunction

logger = logging.getLogger(__name__)


def wolsey_greedy_heuristic(
    budget: float,
    dataset: Dataset,
    separation: Separation,
    submodular_function: SubmodularFunction,
) -> list[str]:
    """Implementation of the Wolsey greedy algorithm

    Args:
        budget (float): Budget threshold
        dataset (Dataset): Dataset
        separation (Separation): Dataset separation object
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

    if heuristic_features:
        while True:
            k += 1

            # Select t_k
            chosen_test = submodular_maximization(
                dataset,
                separation,
                heuristic_features,
                auxiliary_array,
                submodular_function,
            )

            logger.debug("Chosen feature: %s", chosen_test)

            # Remove t_k from T
            heuristic_features.remove(chosen_test)
            # Update spent
            spent += dataset.costs[chosen_test]
            # Update A
            auxiliary_array.append(chosen_test)

            logger.debug(f"Auxiliary array (A): {auxiliary_array}")
            logger.debug("Spent: %f", spent)

            if spent > budget or not heuristic_features:
                break
    else:
        return []

    # Compute f({t_k})
    single_result = submodular_function(dataset, separation, [auxiliary_array[k]])
    # Compute f(A \ {t_k})
    difference_result = submodular_function(dataset, separation, auxiliary_array)

    if single_result >= difference_result:
        # Return {t_k}
        return [auxiliary_array[k]]

    # Return {t_1, ..., t_(k - 1)}
    return dataset.features[:k]
