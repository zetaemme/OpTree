from typing import Callable

from pandas import DataFrame, Series, concat

from src.heuristic import adapted_greedy
from src.pairs import Pairs
from src.utils import evaluate


def calculate_cost(test: Series) -> int:
    """Calculates the cost of a given test

    Parameters
    ----------
    test: The test of which we want to calculate the cost

    Returns
    -------
    int: The cost of the given test
    """
    # FIXME: Implement a real cost function
    return 1


def find_budget(
        objects: DataFrame,
        tests: list[str],
        classes: set[str],
        cost_fn: Callable[[Series], int],
        dataset_pairs_number: int
) -> int:
    """Implementation of the FindBudget procedure of the referenced paper

    Parameters
    ----------
    objects: The dataset we want to classify
    tests: The list of the tests that can be applied to the dataset
    classes: A set containing all the possible classes in the dataset
    cost_fn: A function that computes the effective cost of a given test
    dataset_pairs_number: The number of pairs in the whole dataset

    Returns
    -------
    int: The maximum budget that the algorithm can use to build the Decision Tree
    """

    def submodular_f1(sub_tests: list[str]):
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

            heuristic_test_list = adapted_greedy(objects, tests, submodular_f1, calculate_cost, mid)

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

    return heuristic_binary_search(1, sum([cost_fn(objects[test]) for test in tests]))
