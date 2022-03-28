from pandas import DataFrame

from src.dectree.test import Test
from src.heuristic import adapted_greedy
from src.pairs import Pairs


def calculate_cost(test: Test) -> int:
    """Calculates the cost of a given test

    Args:
        test (Test): The test of which we want to calculate the cost

    Returns:
        int: The cost of the given test
    """
    # FIXME: This value is returned for testing purpose, a real cost function must be created in the future
    return 1


def find_budget(
        objects: DataFrame,
        tests: list[Test],
        test_costs: dict[Test, int],
        classes: set[str],
        # NOTE: By computing the test cost at the beginning of the procedure, we can achieve constant lookup time
        #       extracting the cost of a certain test.
        #       This holds on the assumption that the cost of a test can't change during the execution.
        # cost_fn: Callable[[Test], int],
        dataset_pairs_number: int
) -> int:
    """Implementation of the FindBudget procedure of the referenced paper

    Args:
        objects (DataFrame): The dataset we want to classify
        tests (list[Test]): The list of the tests that can be applied to the dataset
        test_costs (dict[Test, int]): A dictionary containing, for each test, the corresponding effective cost
        classes (set[str]): A set containing all the possible classes in the dataset
        cost_fn (Callable[[Test], int]): A function that computes the effective cost of a given test
        dataset_pairs_number (int): The number of pairs in the whole dataset

    Returns:
        int: The maximum budget that the algorithm can use to build the Decision Tree
    """

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

            heuristic_test_list = adapted_greedy(tests, test_costs, submodular_f1, mid)

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

    return heuristic_binary_search(1, sum([test_costs[test] for test in tests]))
