from typing import Callable

from pandas import DataFrame

from src.dectree.test import Test
from src.heuristic import adapted_greedy
from src.pairs import Pairs


def calculate_cost(test: Test) -> int:
    """Calculates the cost of a given test"""
    # FIXME: This value is returned for testing purpose, create a real cost function
    return 1


def find_budget(
        objects: DataFrame,
        tests: list[Test],
        classes: set[str],
        cost_fn: Callable[[Test], int],
        dataset_pairs_number: int
) -> int:
    """Implementation of the FindBudget procedure of the referenced paper"""

    def submodular_f1(sub_tests: list[Test]):
        items_separated_by_test = [
            item
            for test in sub_tests
            for class_index, _ in enumerate(classes)
            for item in test.evaluate_dataset_for_class(objects, class_index)
        ]

        items_separated_by_test = set(items_separated_by_test)

        sep_pairs = Pairs(DataFrame(
            data=items_separated_by_test,
            columns=objects.columns
        ))

        return dataset_pairs_number - sep_pairs.number

    # NOTE: In the original paper alpha is marked as 1 - e^{X}, approximated with 0.35
    alpha = 0.35

    def heuristic_binary_search(lower, upper) -> int:
        if upper >= lower:
            mid = lower + (upper - lower) / 2

            heuristic_test_list = adapted_greedy(tests, submodular_f1, cost_fn, mid)

            heuristic_test_coverage_sum = sum([
                len(test.evaluate_dataset_for_class(objects, class_index))
                for class_index in range(len(classes))
                for test in heuristic_test_list
            ])

            if heuristic_test_coverage_sum == (alpha * dataset_pairs_number):
                return mid
            elif heuristic_test_coverage_sum > (alpha * dataset_pairs_number):
                return heuristic_binary_search(lower, mid - 1)
            else:
                return heuristic_binary_search(mid + 1, upper)
        else:
            raise ValueError

    return heuristic_binary_search(1, sum([calculate_cost(test) for test in tests]))
