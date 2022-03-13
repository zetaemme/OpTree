from typing import TypeAlias

from src.cost import calculate_cost
from src.data.dataset import Dataset
from src.dectree.test import Test

TestList: TypeAlias = list[Test]


def cheapest_test(tests: TestList) -> Test:
    """Extracts the cheapest (separation cost) test that separates the two objects"""
    if len(tests) == 1:
        return tests[0]

    if all(calculate_cost(test) == 1 for test in tests):
        return tests[0]

    # TODO: Add a return statement for the effective test costs


def object_class(dataset: Dataset, index: int) -> str:
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
