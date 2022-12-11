import logging

from src.dataset import Dataset
from src.separation import Separation

logger = logging.getLogger(__name__)


def submodular_function_1(
    dataset: Dataset, separation: Separation, features: list[str]
) -> int:
    """Submodular function used by the heuristic

    Args:
        dataset (Dataset): The dataset on which the decision is being built
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
