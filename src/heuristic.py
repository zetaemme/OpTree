import logging
from pprint import pformat

from src.dataset import Dataset
from src.maximization import submodular_maximization
from src.types import SubmodularFunction

logger = logging.getLogger("decision_tree")


def wolsey_greedy_heuristic(
        budget: float,
        dataset: Dataset,
        tests: list[str],
        costs: dict[str, float],
        submodular_function: SubmodularFunction,
) -> list[str]:
    """Implementation of the Wolsey greedy algorithm

    Args:
        budget (float): Budget threshold
        dataset (Dataset): Dataset
        tests (list[str]): The tests for the given dataset
        costs (dict[str, float]): The costs for the tests
        submodular_function (SubmodularFunction): Submodular function to apply

    Returns:
        list[str]: A set of features
    """
    logger.info("Applying heuristic for budget: %f", budget)

    spent = 0.0
    auxiliary_array: list[str] = []
    k = -1

    heuristic_features = [
        feature for feature in tests if costs[feature] <= budget
    ]

    if len(heuristic_features) > 0:
        while True:
            k += 1

            # Select t_k
            chosen_test = submodular_maximization(
                dataset,
                costs,
                heuristic_features,
                auxiliary_array,
                submodular_function,
            )
            logger.debug("Test that maximizes the submodular function: %s", chosen_test)

            logger.debug("Removing t_k from T and adding it to A")
            auxiliary_array.append(heuristic_features.pop(heuristic_features.index(chosen_test)))
            logger.debug(f"New A list: {pformat(auxiliary_array)}")

            # Update spent
            spent += costs[chosen_test]
            logger.debug("Adding cost of \"%s\" to spent. Total spent: %d", chosen_test, spent)

            if spent > budget or len(heuristic_features) == 0:
                break
    else:
        return []

    # Compute f({t_k})
    single_result = submodular_function(dataset, [auxiliary_array[k]])
    logger.debug("f({t_k}): %i", single_result)

    # Compute f(A \ {t_k})
    # NOTE: We can use [:-1] since t_K is the last element of the list
    difference_result = submodular_function(dataset, auxiliary_array[:-1])
    logger.debug("f(A \\ {t_k}): %i", difference_result)

    if single_result >= difference_result:
        # Return {t_k}
        logger.debug("Returning \"%s\"", auxiliary_array[k])
        return [auxiliary_array[k]]

    # Return {t_1, ..., t_(k - 1)}
    logger.debug(f"Returning {pformat(auxiliary_array[:k])}")
    return auxiliary_array[:k]
