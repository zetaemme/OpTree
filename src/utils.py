import logging

from src.dataset import Dataset
from src.separation import Separation

logger = logging.getLogger(__name__)


def submodular_function_1(dataset: Dataset, features: list[str]) -> int:
    """Submodular function used by the heuristic

    Args:
        dataset (Dataset): The dataset on which the decision is being built
        features (list[str]): The subset of features to consider

    Returns:
        int: The difference between the dataset "Pairs" and the number of pairs in the S_star[feature] intersection
    """
    submodular_separation = Separation(dataset.from_features_subset(features))

    if len(submodular_separation.S_star_intersection) == 1:
        return dataset.pairs_number

    return dataset.pairs_number - dataset.pairs_number_for(submodular_separation.S_star_intersection)
