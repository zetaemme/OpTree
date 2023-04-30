import logging

from src.dataset import Dataset
from src.types import SubmodularFunction

logger = logging.getLogger("decision_tree")


def probability_maximization(
        universe: Dataset,
        tests: list[str],
        costs: dict[str, float],
        budget: float,
        spent: float
) -> str:
    def calculate_probability_maximization_for(feature: str) -> float:
        intersection = universe.intersection(universe.S_star[feature])
        return (universe.total_probability - intersection.total_probability) / costs[feature]

    maximum_eligible = {
        feature: calculate_probability_maximization_for(feature)
        for feature in tests
        # NOTE: Not sure if second condition works or even if it's something that can happen
        if costs[feature] <= budget - spent and feature in universe.S_star.keys()
    }

    return max(maximum_eligible, key=maximum_eligible.get)  # type: ignore


def pairs_maximization(universe: Dataset, tests: list[str], costs: dict[str, float]) -> str:
    def calculate_pairs_maximization_for(feature: str) -> float:
        intersection = universe.intersection(universe.S_star[feature])
        return (universe.pairs_number - intersection.pairs_number) / costs[feature]

    maximum_eligible = {
        feature: calculate_pairs_maximization_for(feature)
        for feature in tests
        # NOTE: Not sure if second condition works or even if it's something that can happen
        if feature in universe.S_star.keys()
    }

    return max(maximum_eligible, key=maximum_eligible.get)  # type: ignore


def submodular_maximization(
        dataset: Dataset,
        costs: dict[str, float],
        heuristic_features: list[str],
        auxiliary_features: list[str],
        submodular_function: SubmodularFunction,
) -> str:
    logger.debug(f"Maximizing submodular function in {heuristic_features}")

    maximum_eligible: dict[str, float] = {}
    for feature in heuristic_features:
        logger.debug("Feature: %s", feature)

        # Computes f(A)
        feature_result = submodular_function(dataset, auxiliary_features)
        logger.debug("f(A): %i", feature_result)

        # Computes f(A U {t})
        union_result = submodular_function(dataset, auxiliary_features + [feature])
        logger.debug("f(A U {t}): %i", union_result)

        # NOTE: 22/03/2023 - We discovered a typo in the original paper.
        #       To avoid issues with the budget the following was changed:
        #           (f(A U {t}) - f(A))  ->  (f(A) - f(A U {t}))
        #       Doing so, we avoid negative numbers in maximum_eligible
        submodular_result = (feature_result - union_result) / costs[feature]
        logger.debug("(f(A) - f(A U {t})) / %f = %f", costs[feature], submodular_result)
        maximum_eligible[feature] = submodular_result

    logger.debug(f"\n{pformat(maximum_eligible)}")
    return max(maximum_eligible, key=maximum_eligible.get)
