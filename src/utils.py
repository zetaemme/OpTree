from numpy import ndarray, take

from src.dataset import Dataset
from src.separation import Separation


def submodular_function_1(dataset: Dataset, features: list[str]) -> int:
    """Submodular function used by the heuristic

    Args:
        dataset (Dataset): The dataset on which the decision is being built
        features (list[str]): The subset of features to consider

    Returns:
        int: The difference between the dataset "Pairs" and the number of pairs in the S_star[feature] intersection
    """
    submodular_separation = Separation(dataset.from_features_subset(features))

    max_intersection: ndarray = submodular_separation.S_star_intersection

    if len(max_intersection) == 1:
        return dataset.pairs_number

    intersection_pairs = Dataset.Pairs(
        take(dataset.data(complete=True), max_intersection, axis=0)
    )

    return dataset.pairs_number - intersection_pairs.number
