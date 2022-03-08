from typing import Callable, TypeAlias

from pandas import DataFrame

from src.dectree.test import Test
from src.pairs import Pairs

TestList: TypeAlias = list[Test]


def calculate_cost(test: Test) -> int:
    """Calculates the cost of a given test"""
    # FIXME: Settare a vera funzione di costo, 1 messo solo per testing
    return 1


def find_budget(objects: DataFrame, tests: list[Test], classes: set[str], cost_fn: Callable[[Test], int]) -> int:
    """Implementation of the FindBudget procedure of the referenced paper"""

    def submodular_f1(sub_tests: TestList):
        pairs = Pairs(objects)

        items_separated_by_test = [
            item
            for test in sub_tests
            for index, _ in enumerate(classes)
            for item in test.evaluate_dataset_for_class(objects, index)
        ]

        items_separated_by_test = list(set(items_separated_by_test))

        sep_pairs = Pairs(DataFrame(items_separated_by_test))

        return pairs.number - sep_pairs.number

    alpha = 0.35

    # TODO: Continuare da "Do a binary search..."
    return 0
