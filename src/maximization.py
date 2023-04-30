import logging
from pprint import pformat

from src.dataset import Dataset
from src.types import SubmodularFunction

logger = logging.getLogger("decision_tree")


def probability_maximization(universe: Dataset, tests: list[str], costs: dict[str, float]) -> str:
    def submodular_probabilities() -> dict[str, float]:
        logger.debug("Computing U ∩ S[*][feature] for each feature:")
        per_feature_intersection_probability = {
            feature: universe.intersection(universe.S_star[feature]).total_probability
            for feature in tests
        }
        logger.debug(pformat(per_feature_intersection_probability))

        return {
            feature: (universe.total_probability - per_feature_intersection_probability[feature]) / costs[feature]
            for feature in tests
            if feature in universe.S_star.keys()
        }

    return max(submodular_probabilities(), key=submodular_probabilities().get)


def pairs_maximization(universe: Dataset, tests: list[str], costs: dict[str, float]) -> str:
    def submodular_pairs() -> dict[str, float]:
        logger.debug("Computing U ∩ S[*][feature] for each feature:")
        per_feature_intersection_pairs = {
            feature: universe.intersection(universe.S_star[feature]).pairs_number
            for feature in tests
        }

        return {
            feature: (universe.pairs_number - per_feature_intersection_pairs[feature]) / costs[feature]
            for feature in tests
            if feature in universe.S_star.keys()
        }

    return max(submodular_pairs(), key=submodular_pairs().get)


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
