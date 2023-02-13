import logging

from src.dataset import Dataset
from src.types import SubmodularFunction

logger = logging.getLogger(__name__)


def probability_maximization(universe: Dataset, budget: float, spent: float) -> str:
    universe_copy = universe.copy()

    def calculate_probability_maximization_for(feature: str) -> float:
        intersection = universe_copy.intersection(universe_copy.S_star[feature])
        return (universe.total_probability - intersection.total_probability) / universe.costs[feature]

    maximum_eligible: dict[str, float] = {
        feature: calculate_probability_maximization_for(feature)
        for feature in universe.features
        if universe.costs[feature] <= budget - spent
    }

    return max(maximum_eligible, key=maximum_eligible.get)  # type: ignore


def pairs_maximization(universe: Dataset) -> str:
    universe_copy = universe.copy()

    def calculate_pairs_maximization_for(feature: str) -> float:
        intersection = universe_copy.intersection(universe_copy.S_star[feature])
        return (universe.pairs_number - intersection.pairs_number) / universe.costs[feature]

    maximum_eligible: dict[str, float] = {
        feature: calculate_pairs_maximization_for(feature) for feature in universe.features
    }

    return max(maximum_eligible, key=maximum_eligible.get)  # type: ignore


def submodular_maximization(
    dataset: Dataset,
    heuristic_features: list[str],
    auxiliary_features: list[str],
    submodular_function: SubmodularFunction,
) -> str:
    logger.info(f"Maximizing submodular function in {heuristic_features}")

    maximum_eligible: dict[str, float] = {}
    for feature in heuristic_features:
        logger.debug("Feature: %s", feature)

        # Computes f(A)
        feature_result = submodular_function(dataset, auxiliary_features)
        logger.debug("f(A): %i", feature_result)

        # Computes f(A U {t})
        union_result = submodular_function(dataset, auxiliary_features + [feature])
        logger.debug("f(A U {t}): %i", union_result)

        submodular_result = (union_result - feature_result) / dataset.costs[feature]
        maximum_eligible[feature] = submodular_result

    return max(maximum_eligible, key=maximum_eligible.get)  # type: ignore
