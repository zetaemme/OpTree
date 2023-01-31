import logging

from src.dataset import Dataset
from src.separation import Separation
from src.types import Bounds, HeuristicFunction

from itertools import chain

logger = logging.getLogger(__name__)


def submodular_function_1(dataset: Dataset, separation: Separation, features: list[str]) -> int:
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

    if len(submodular_separation.S_star_intersection) < 2:
        return dataset.pairs_number

    return dataset.pairs_number - dataset.pairs_number_for(submodular_separation.S_star_intersection)


def binary_search_budget(
    dataset: Dataset,
    separation: Separation,
    search_range: Bounds,
    heuristic: HeuristicFunction,
) -> float:
    """Calculates the procedure's budget via Binary Search

    Args:
        dataset (Dataset): The dataset on which the decision is being built
        separation (Separation): Dataset tripartition and sets
        search_range (list[float]): Range in which the binary search is performed
        heuristic (HeuristicFunction): Heuristic function

    Returns:
        float: The optimal budget for the procedure
    """
    result = 0.0

    # Should be (1 - e^{chi}), approximated with 0.35 in the paper
    alpha = 0.35

    while search_range.upper >= search_range.lower + 1:
        current_budget = (search_range.lower + search_range.upper) / 2

        heuristic_result = heuristic(current_budget, dataset, separation, submodular_function_1)

        logger.debug(f"Heuristic result: {heuristic_result}")

        covered_pairs = [set(separation.kept[test] + separation.separated[test]) for test in heuristic_result]
        covered_pairs = set(chain(*covered_pairs))

        logger.debug(f"Pairs covered by the heuristic: {covered_pairs}")

        if len(covered_pairs) < (alpha * dataset.pairs_number):
            search_range.upper = current_budget
        else:
            search_range.lower = current_budget

        result = current_budget

    return result
