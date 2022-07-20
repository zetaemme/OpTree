from typing import Callable

from pandas import DataFrame, Series


def adapted_greedy(
        objects: DataFrame,
        tests: list[str],
        submodular_f: Callable[[list[str]], int],
        cost_fn: Callable[[Series], int],
        budget: int | float
) -> list[str]:
    """Implementation of the Adapted-Greedy heuristic

    Parameters
    ----------
    objects: DataFrame
        The dataset
    tests: list[str]
        A list containing all the tests
    submodular_f: Callable[[list[str]], int]
        A submodular function
    cost_fn: Callable[[Series], int]
        A function that calculates the effective cost of a given test
    budget: int
        The maximum budget that a test list should not cross

    Returns
    -------
    tests_sublist: list[str]
        The maximum list of tests that can be used without crossing the budget
    """
    spent = 0
    A = []
    k = -1

    # Remove from T all tests with cost larger than B
    tests_with_ariety = {ariety: test for ariety, test in enumerate(
        [test for test in tests if cost_fn(objects[test]) <= budget]
    )}

    # Repeat until spent > budget or test becomes empty
    while spent <= budget and tests_with_ariety:
        k += 1

        # Removes and returns the k-th item from the test list
        minimizing_test_key = min(tests_with_ariety, key=tests_with_ariety.get)
        minimizing_test_value = tests_with_ariety[minimizing_test_key]

        del tests_with_ariety[minimizing_test_key]

        # Calculate the cost of the k-th test and add it to 'spent'
        cost = cost_fn(objects[minimizing_test_value])
        spent += cost

        A.append(minimizing_test_value)

    # If the k-th test covers more _pairs than all the others, return the k-th test
    if submodular_f([tests[k]]) >= submodular_f([a for a in A if a != tests[k]]):
        return [tests[k]]

    # Otherwise, return all the test before the k-th
    return tests[:k]
