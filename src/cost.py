from typing import Callable

from pandas import DataFrame, Series

from src.heuristic import adapted_greedy
from src.pairs import Pairs
from src.utils import evaluate


def calculate_cost(test: Series) -> int:
    """Calculates the cost of a given test

    Parameters
    ----------
    test: Series
        The test of which we want to calculate the cost

    Returns
    -------
    cost: int
        The cost of the given test
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
    objects: DataFrame
        The dataset we want to classify
    tests: list[str]
        The list of the tests that can be applied to the dataset
    classes: set[str]
        A set containing all the possible classes in the dataset
    cost_fn: Callable[[Series], int]
        A function that computes the effective cost of a given test
    dataset_pairs_number: int
        The number of pairs in the whole dataset

    Returns
    -------
    budget: int
        The maximum budget that the algorithm can use to build the Decision Tree
    """

    def submodular_f1(sub_tests: list[str]) -> int:
        return dataset_pairs_number - Pairs(evaluate.dataframe_intersection([
            evaluate.maximum_separation_set_for_test(objects, test) for test in sub_tests
        ])).number

    # NOTE: In the original paper alpha is marked as (1 - e^{X}), approximated with 0.35
    alpha = 0.35

    def heuristic_binary_search(lower, upper) -> int:
        if upper >= lower:
            mid = lower + (upper - lower) / 2

            heuristic_test_list = adapted_greedy(objects, tests, submodular_f1, calculate_cost, mid)

            kept_df = DataFrame()
            separated_df = DataFrame()

            for test in heuristic_test_list:
                kept_df.concat(evaluate.objects_kept_by_test(objects, test)).drop_duplicates()

            for test in heuristic_test_list:
                separated_df.concat(evaluate.objects_separated_by_test(objects, test)).drop_duplicates()

            covering = max(kept_df.shape[0], separated_df.shape[0])

            if covering == (alpha * dataset_pairs_number):
                return mid
            elif covering > (alpha * dataset_pairs_number):
                return heuristic_binary_search(lower, mid - 1)
            else:
                return heuristic_binary_search(mid + 1, upper)
        else:
            raise ValueError

    return heuristic_binary_search(1, sum([cost_fn(objects[test]) for test in tests]))
