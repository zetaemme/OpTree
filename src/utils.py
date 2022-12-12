import logging
from typing import Callable

from src.dataset import Dataset
from src.separation import Separation

logger = logging.getLogger(__name__)


def submodular_function_1(
    dataset: Dataset, separation: Separation, features: list[str]
) -> int:
    """Submodular function used by the heuristic

    Args:
        dataset (Dataset): The dataset on which the decision is being built
        separation (Separation): Dataset tripartition and sets
        features (list[str]): The subset of features to consider

    Returns:
        int: The difference between the dataset "Pairs" and the number of pairs in the S_star[feature] intersection
    """
    if not features:
        return dataset.pairs_number

    submodular_separation = separation.for_features_subset(features)

    if len(submodular_separation.S_star_intersection) == 1:
        return dataset.pairs_number

    return dataset.pairs_number - dataset.pairs_number_for(
        submodular_separation.S_star_intersection
    )


def get_parent_node(features: list[str], child: str) -> str:
    """Gets the name of child's parent node

    Args:
        features (list[str]): List of all possible nodes
        child (str): Node that we want to connect with his parent

    Returns:
        str: The name of the parent node
    """
    parent_index = features.index(child) - 1
    return features[parent_index]


def binary_search_budget(
    dataset: Dataset,
    separation: Separation,
    search_range: list[float],
    heuristic: Callable[
        [
            float,
            Dataset,
            Separation,
            Callable[[Dataset, Separation, list[str]], int],
        ],
        list[str],
    ],
) -> float:
    """Calculates the procedure's budget via Binary Search

    Args:
        dataset (Dataset): The dataset on which the decision is being built
        separation (Separation): Dataset tripartition and sets
        search_range (list[float]): Range in which the binary search is performed
        heuristic (Callable): Heuristic function

    Returns:
        float: The optimal budget for the procedure
    """
    result = 0.0

    # Should be (1 - e^{chi}), approximated with 0.35 in the paper
    alpha = 0.35
    while search_range[0] <= search_range[1]:
        mid = (search_range[0] + search_range[1]) / 2

        heuristic_result = heuristic(mid, dataset, separation, submodular_function_1)

        covered_pairs = list(
            set(separation.kept[test] + separation.separated[test])
            for test in heuristic_result
        )

        if len(covered_pairs) < (alpha * dataset.pairs_number):
            search_range[0] = mid + 1
        else:
            result = mid
            break

    return result
