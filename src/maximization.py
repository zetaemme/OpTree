import logging

from src.dataset import Dataset
from src.separation import Separation
from src.types import SubmodularFunction

logger = logging.getLogger(__name__)


def probability_maximization(universe: Dataset, budget: float, spent: float) -> str:
    universe_separation = Separation(universe)

    def calculate_probability_maximization_for(feature: str) -> float:
        intersection = universe.intersection(universe_separation.S_star[feature])
        return (
            universe.total_probability - intersection.total_probability
        ) / universe.costs[feature]

    maximum_eligible: dict[str, float] = {
        feature: calculate_probability_maximization_for(feature)
        for feature in universe.features
        if universe.costs[feature] <= budget - spent
    }

    return max(maximum_eligible, key=maximum_eligible.get)


def pairs_maximization(universe: Dataset) -> str:
    universe_separation = Separation(universe)

    def calculate_pairs_maximization_for(feature: str) -> float:
        intersection = universe.intersection(universe_separation.S_star[feature])
        return (universe.pairs_number - intersection.pairs_number) / universe.costs[
            feature
        ]

    maximum_eligible: dict[str, float] = {
        feature: calculate_pairs_maximization_for(feature)
        for feature in universe.features
    }

    return max(maximum_eligible, key=maximum_eligible.get)


def submodular_maximization(
    dataset: Dataset,
    separation,
    features: list[str],
    submodular_function: SubmodularFunction,
) -> str:
    logger.info(f"Maximizing submodular function for {features}")
    maximum_eligible: dict[str, float] = {}
    for feature in dataset.features:
        feature_result = submodular_function(dataset, separation, features)
        features.append(feature)
        union_result = submodular_function(dataset, separation, features)
        features.remove(feature)

        submodular_result = (union_result - feature_result) / dataset.costs[feature]

        maximum_eligible[feature] = submodular_result

    return max(maximum_eligible, key=maximum_eligible.get)
