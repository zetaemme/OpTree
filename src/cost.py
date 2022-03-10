from typing import Callable, TypeAlias

from src.data.dataset import Dataset
from src.dectree.test import Test
from src.heuristic import adapted_greedy
from src.pairs import Pairs

TestList: TypeAlias = list[Test]


def calculate_cost(test: Test) -> int:
    """Calculates the cost of a given test"""
    # FIXME: Settare a vera funzione di costo, 1 messo solo per testing
    return 1


def find_budget(objects: Dataset, tests: list[Test], classes: set[str], cost_fn: Callable[[Test], int]) -> int:
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

        sep_pairs = Pairs(Dataset(data=items_separated_by_test, columns=None))

        return pairs.number - sep_pairs.number

    # NOTE: In the original paper this is marked as 1 - e^{X}, approximated with 0.35
    alpha = 0.35

    pairs = Pairs(objects)

    # FIXME: Dovrebbe essere eseguito in BinarySearch, invece che linearmente
    for b in range(1, sum([calculate_cost(test) for test in tests]) + 1):
        heuristic_test_list = adapted_greedy(tests, submodular_f1, cost_fn, b)

        heuristic_test_coverage_sum = 0
        for test in heuristic_test_list:
            for index in range(len(classes)):
                heuristic_test_coverage_sum += len(test.evaluate_dataset_for_class(objects, index))

        if heuristic_test_coverage_sum >= (alpha * pairs.number):
            return b
