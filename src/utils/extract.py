from pandas import DataFrame

from src.cost import calculate_cost
from src.dectree.test import Test
from src.pairs import Pairs


def cheapest_test(tests: list[Test]) -> Test:
    """Extracts the cheapest (separation cost) test that separates the two objects"""
    if len(tests) == 1:
        return tests[0]

    if all(calculate_cost(test) == 1 for test in tests):
        return tests[0]

    # TODO: Add a return statement for the effective test costs, since the calculate_cost function always returns 1.
    #       Connected to the future implementation of a real calculate_cost(...) function


def maximum_separated_class(
        items_separated_by_test: dict[Test, DataFrame],
        maximizing_test: Test,
        classes: set[str]
) -> DataFrame:
    """Extracts the set S^{*}_{maximizing_test}"""
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
    """Extracts the class label from the item in position index of a given dataset"""
    assert index >= 0, "Index should be a positive integer"
    return dataset.rows[index]['class']


def test_structure(test: str) -> Test:
    """Extracts the test structure (lhs, type, rhs) from a given string"""
    structure = test.split()

    if float(structure[2]).is_integer():
        rhs = int(structure[2])
    else:
        rhs = float(structure[2])

    return Test(structure[0], structure[1], rhs, list(map(int, structure[3:])))


def tests_costing_less_than(tests: list[Test], cost: int) -> list[Test]:
    """Extracts all the tests which cost is less than a given cost"""
    # NOTE: Doing this assignment avoids the case in which a Generator is returned instead of a list
    result = [test for test in tests if calculate_cost(test) <= cost]
    return result
