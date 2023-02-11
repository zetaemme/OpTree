from src.dataset import Dataset
from src.separation import Separation


def cheapest_separation(dataset: Dataset, separation: Separation, obj_idx1: int, obj_idx2: int, ) -> str:
    for feature, _ in sorted(dataset.costs.items(), key=lambda x: x[1]):
        if tuple([obj_idx1, obj_idx2]) in separation.separated[feature]:
            return feature


def eligible_labels(dataset: Dataset, separation: Separation, test: str) -> list[str]:
    eligible = []

    for label in dataset.labels_for(test):
        inter = dataset.intersection(separation.S_label[test][label])
        if inter:
            eligible.append(label)

    if len(eligible) == 0:
        return []

    return [label for label in eligible if separation.S_label[test][label] != separation.S_star[test]]
