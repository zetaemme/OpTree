from src.dataset import Dataset


def cheapest_separation(dataset: Dataset, obj_idx1: int, obj_idx2: int, ) -> str:
    for feature, _ in sorted(dataset.costs.items(), key=lambda x: x[1]):
        if tuple([obj_idx1, obj_idx2]) in dataset.separated[feature]:
            return feature


def eligible_labels(dataset: Dataset, test: str) -> list[str]:
    eligible = []

    for label in dataset.labels_for(test):
        inter = dataset.intersection(dataset.S_label[test][label])
        if inter:
            eligible.append(label)

    if len(eligible) == 0:
        return []

    return [label for label in eligible if dataset.S_label[test][label] != dataset.S_star[test]]
