from typing import Callable

from pandas import DataFrame, Series

from src.pairs import Pairs


def cheapest_test(objects: DataFrame, tests: list[str], cost_fn: Callable[[Series], int]) -> str:
    """Extracts the cheapest (separation cost) test that separates the two objects

    Parameters
    ----------
    objects: The dataset containing the objects to classify
    tests: The list from which the cheapest test will be extracted
    cost_fn: A function returning the effective cost of a given test

    Returns
    -------
    str: The minimum cost test in the tests list
    """
    if len(tests) == 1:
        return tests[0]

    if all(cost_fn(objects[test]) == 1 for test in tests):
        return tests[0]

    # FIXME: Not tested
    return min({test: cost_fn(objects[test]) for test in tests}, key=dict.get)


def maximum_separated_class(
        items_separated_by_test: dict[str, DataFrame],
        maximizing_test: str,
        classes: set[str]
) -> DataFrame:
    """Extracts the set S^{*}_{maximizing_test} from a given dictionary of separated objects

    Parameters
    ----------
    items_separated_by_test: The dictionary containing, for each test, a DataFrame of all the objects separated from a
                             specific test
    maximizing_test: The test t for which we want to calculate the S^{*}_{t} set
    classes: A set containing all the classes in the dataset

    Returns
    -------
    DataFrame: A Pandas DataFrame representing the S^{*}_{t} set
    """
    separation_list = {
        items_separated_by_test[maximizing_test][class_label]:
            Pairs(items_separated_by_test[maximizing_test][class_label])
        for class_label in classes
    }

    # Extracts the target pair value for S^{*}_{t_k}
    max_pair_number = max([pair.number for pair in separation_list.values()])

    maximum_separated_class_from_tk = None

    for separation_set in separation_list.items():
        if separation_set[1].number == max_pair_number:
            # NOTE: Corresponds to S^{*}_{t_k}
            maximum_separated_class_from_tk = separation_set[0]

    assert maximum_separated_class_from_tk is not None
    return maximum_separated_class_from_tk


def object_class(dataset: DataFrame, index: int) -> str:
    """Extracts the class label from the item in position index of a given dataset

    Parameters
    ----------
    dataset: The set of objects containing the object in position the given position
    index: The index of the item of which we want to extract the class

    Returns
    -------
    str: A string representing the objects class
    """
    assert index >= 0, "Index should be a positive integer"
    return dataset.rows[index]['class']


def tests_costing_less_than(
        objects: DataFrame,
        tests: list[str],
        cost_fn: Callable[[Series], int],
        cost: int
) -> list[str]:
    """Extracts all the tests which cost is less than a given cost

    Parameters
    ----------
    objects: The dataset containing the objects to classify
    tests: The list in which we need to search
    cost_fn: A function returning the effective cost of a given test
    cost: The threshold we mustn't cross

    Returns
    -------
    list[Test]: A list containing all tests of effective cost less than the given cost
    """
    # NOTE: Doing this assignment avoids the case in which a Generator is returned instead of a list
    result = [test for test in tests if cost_fn(objects[test]) <= cost]
    return result
