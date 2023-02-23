import logging
from itertools import chain
from re import match
from typing import Any

from src.dataset import Dataset
from src.types import Bounds, HeuristicFunction

logger = logging.getLogger(__name__)


def submodular_function_1(dataset: Dataset, features: list[str]) -> int:
    """Submodular function used by the heuristic

    Args:
        dataset (Dataset): The dataset on which the decision is being built
        features (list[str]): The subset of features to consider

    Returns:
        int: The difference between the dataset "Pairs" and the number of pairs in the S_star[feature] intersection
    """
    if not features:
        return dataset.pairs_number

    submodular_separation = dataset.separation_for_features_subset(features)

    if len(submodular_separation.S_star_intersection) < 2:
        return dataset.pairs_number

    return dataset.pairs_number - dataset.pairs_number_for(submodular_separation.S_star_intersection)


def binary_search_budget(
    dataset: Dataset,
    search_range: Bounds,
    heuristic: HeuristicFunction,
) -> float:
    """Calculates the procedure's budget via Binary Search

    Args:
        dataset (Dataset): The dataset on which the decision is being built
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

        heuristic_result = heuristic(current_budget, dataset, submodular_function_1)

        logger.debug(f"Heuristic result: {heuristic_result}")

        covered_pairs = [set(dataset.kept[test] + dataset.separated[test]) for test in heuristic_result]
        covered_pairs = set(chain(*covered_pairs))

        logger.debug(f"Pairs covered by the heuristic: {covered_pairs}")

        if len(covered_pairs) < (alpha * dataset.pairs_number):
            search_range.upper = current_budget
        else:
            search_range.lower = current_budget

        result = current_budget

    return result


def parse(value: dict[str, list]) -> dict[Any, list]:
    def cast(key: str) -> int | float:
        if bool(match(r"^(-)?\d+.\d+$", key)):
            return float(key)
        else:
            return int(key)

    return {cast(key): val for key, val in value.items()}


def get_backbone_label(dataset: Dataset, feature: str) -> str:
    for key, value in dataset.S_label[feature].items():
        if value == dataset.S_star[feature]:
            return key
