from typing import Callable, NamedTuple

from pandas import DataFrame, Series

from src.pairs import Pairs


def cheapest_test(objects: DataFrame, tests: list[str], cost_fn: Callable[[Series], int]) -> str:
    """Extracts the cheapest (separation cost) test that separates the two objects

    Parameters
    ----------
    objects: DataFrame
        The dataset containing the objects to classify
    tests: list[str]
        The list from which the cheapest test will be extracted
    cost_fn: Callable[[Series], int]
        A function returning the effective cost of a given test

    Returns
    -------
    min_cost_test: str
        The minimum costing test in the tests list
    """
    if len(tests) == 1:
        return tests[0]

    if all(cost_fn(objects[test]) == 1 for test in tests):
        return tests[0]

    return min({test: cost_fn(objects[test]) for test in tests}, key=dict.get)


def maximum_separated_class(
        items_separated_by_test: dict[str, DataFrame],
        maximizing_test: str,
        feature_values: set[str]
) -> DataFrame:
    """Extracts the set S^{*}_{maximizing_test} from a given dictionary of separated objects

    Parameters
    ----------
    items_separated_by_test: dict[str, DataFrame]
        The dictionary containing, for each test, a DataFrame of all the objects separated from a specific test
    maximizing_test: str
        The test t for which we want to calculate the S^{*}_{t} set
    feature_values: set[str]
        A set containing all the classes in the dataset

    Returns
    -------
    maximum_separated_class_from_tk: DataFrame
        A Pandas DataFrame representing the S^{*}_{t} set
    """
    # NOTE: NamedTuple is used instead of dict because DataFrame is not hashable
    SepList = NamedTuple('SepList', [('data_frame', DataFrame), ('pairs', Pairs)])

    separation_list = [
        SepList(
            items_separated_by_test[maximizing_test][str(value)],
            Pairs(items_separated_by_test[maximizing_test][str(value)])
        )
        for value in feature_values
    ]

    # Extracts the target pair value for S^{*}_{t_k}
    # NOTE: Horrible to see, but necessary since what is written in previous NOTE still holds
    max_pair_number = max(separation_list, key=lambda x: x.pairs.number).pairs.number

    maximum_separated_class_from_tk = None

    for separation_set in separation_list:
        if separation_set.pairs.number == max_pair_number:
            # NOTE: Corresponds to S^{*}_{t_k}
            maximum_separated_class_from_tk = separation_set.data_frame

    assert maximum_separated_class_from_tk is not None
    return maximum_separated_class_from_tk


def object_class(dataset: DataFrame, index: int) -> str:
    """Extracts the class label from the item in position index of a given dataset

    Parameters
    ----------
    dataset: DataFrame
        The set of objects containing the object in position the given position
    index: int
        The index of the item of which we want to extract the class

    Returns
    -------
    class: str
        A string representing the objects class
    """
    assert index >= 0, "Index should be a positive integer"
    return dataset['class'][index]


def tests_costing_less_than(
        objects: DataFrame,
        tests: list[str],
        cost_fn: Callable[[Series], int],
        cost: int
) -> list[str]:
    """Extracts all the tests which cost is less than a given cost

    Parameters
    ----------
    objects: DataFrame
        The dataset containing the objects to classify
    tests: str
        The list in which we need to search
    cost_fn: Callable[[Series], int]
        A function returning the effective cost of a given test
    cost: int
        The threshold we mustn't cross

    Returns
    -------
    resulting_tests: list[str]
        A list containing all tests of effective cost less than the given cost
    """
    # NOTE: Doing this assignment avoids the case in which a Generator is returned instead of a list
    resulting_tests = [test for test in tests if cost_fn(objects[test]) <= cost]
    return resulting_tests
