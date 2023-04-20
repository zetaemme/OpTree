from src.dataset import Dataset


def cheapest_separation(dataset: Dataset, costs: dict[str, float], pair: tuple[int, int]) -> str:
    if len(dataset.features) == 1:
        return dataset.features[0]

    separating_tests = [test for test, pairs in dataset.separated.items() if pair in pairs]

    if len(separating_tests) == 1:
        return separating_tests[0]

    separating_dict = {test: cost for test, cost in costs.items() if test in separating_tests}

    return min(separating_dict, key=separating_dict.get)  # type: ignore


def eligible_labels(dataset: Dataset, test: str) -> list[int]:
    eligible = []

    for label in dataset.labels_for(test):
        inter = dataset.intersection(dataset.S_label[test][label])
        if len(inter) != 0:
            eligible.append(label)

    if len(eligible) == 0:
        return []

    return [label for label in eligible if dataset.S_label[test][label] != dataset.S_star[test]]
