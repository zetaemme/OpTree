import logging
from pprint import pformat

from src.dataset import Dataset

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
        auxiliary_features: list[str]
) -> str:
    logger.debug(f"Maximizing submodular function in {heuristic_features}")

    maximum_eligible: dict[str, float] = {}
    for feature in heuristic_features:
        logger.debug("Feature: %s", feature)

        # Computes P(∩ S[*][t for t in A])
        # NOTE: 03/05/2023 - We chose to consider P(∩ S[*][t for t in A]) instead of f(A) since we can algebraically
        #       reduce f(A U {t}) - f(A) to P(∩ S[*][t for t in A]) - P(∩ S[*][t for t in A U {t}])
        non_union_result = dataset.pairs_number_for(dataset.S_star_intersection_for_features(auxiliary_features))
        logger.debug("P(∩ S[*][t for t in A]): %i", non_union_result)

        # Computes P(∩ S[*][t for t in A U {t}])
        # NOTE: 03/05/2023 - We chose to consider P(∩ S[*][t for t in A U {t}]) instead of f(A U {t}) since we can
        #       algebraically reduce f(A U {t}) - f(A) to P(∩ S[*][t for t in A]) - P(∩ S[*][t for t in A U {t}])
        union_result = dataset.pairs_number_for(
            dataset.S_star_intersection_for_features(auxiliary_features + [feature]))
        logger.debug("P(∩ S[*][t for t in A U {t}]): %i", union_result)

        submodular_result = (non_union_result - union_result) / costs[feature]
        logger.debug("(f(A U {t}) - f(A)) / %f = %f", costs[feature], submodular_result)
        maximum_eligible[feature] = submodular_result

    logger.debug(f"\n{pformat(maximum_eligible)}")
    return max(maximum_eligible, key=maximum_eligible.get)
