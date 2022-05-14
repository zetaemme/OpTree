import logging
from typing import Callable

from pandas import DataFrame, Series, concat

from src.heuristic import adapted_greedy
from src.pairs import Pairs
from src.utils import evaluate

logger = logging.getLogger(__name__)

logging.basicConfig(
    filename='dectree.log',
    format='%(levelname)s (%(asctime)s): %(message)s',
    level=logging.INFO
)


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
    return len(test.unique())


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
        maximum_separation_set = [evaluate.maximum_separation_set_for_test(objects, test) for test in sub_tests]

        eval_result = evaluate.dataframe_intersection(maximum_separation_set)

        dataframe_intersection_pairs = Pairs(eval_result)
        return dataset_pairs_number - dataframe_intersection_pairs.number

    # NOTE: In the original paper alpha is marked as (1 - e^{X}), approximated with 0.35
    alpha = 0.35

    def heuristic_binary_search(
            lower: int | float,
            upper: int | float,
            kept_df: DataFrame = DataFrame(),
            separated_df: DataFrame = DataFrame()
    ) -> int | float:
        if upper >= lower:
            mid = lower + (upper - lower) / 2

            heuristic_test_list = adapted_greedy(objects, tests, submodular_f1, calculate_cost, mid)

            for test in heuristic_test_list:
                objects_kept_by_test = evaluate.objects_kept_by_test(objects, test)
                kept_df = concat([kept_df, objects_kept_by_test]).drop_duplicates()

            for test in heuristic_test_list:
                objects_separated_by_test = evaluate.objects_separated_by_test(objects, test)
                separated_df = concat([separated_df, objects_separated_by_test]).drop_duplicates()

            covering = max(kept_df.shape[0], separated_df.shape[0])

            # FIXME: Se mi basta che copra almeno (alpha * dataset_pairs_number) paia, che Binary Search è?
            # if covering > (alpha * dataset_pairs_number):
            #     print(f'{mid=} {covering=}')
            #     return heuristic_binary_search(lower, mid - 1)

            if covering < (alpha * dataset_pairs_number):
                return heuristic_binary_search(mid + 1, upper)

            return mid
        else:
            logger.error('No budget found during Binary Search!')
            raise ValueError

    return heuristic_binary_search(1, sum([cost_fn(objects[test]) for test in tests]))
