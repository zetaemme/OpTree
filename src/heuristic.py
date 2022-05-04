from typing import Callable

from pandas import DataFrame, Series


def adapted_greedy(
        objects: DataFrame,
        tests: list[str],
        f: Callable,
        cost_fn: Callable[[Series], int],
        budget: int
) -> list[str]:
    """Implementation of the Adapted-Greedy heuristic

    Parameters
    ----------
    objects: DataFrame
        The dataset
    tests: list[str]
        A list containing all the tests
    f: Callable
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
    assert budget >= 0, 'Bound should be a positive integer!'

    spent = 0
    A = []
    k = 0

    # Remove from T all tests with cost larger than B
    tests = [test for test in tests if cost_fn(objects[test]) <= budget]

    if not tests:
        while True:
            k += 1

            # Removes and returns the k-th item from the test list
            tk = tests.pop(k)

            # Calculate the cost of the k-th test and add it to 'spent'
            cost = cost_fn(objects[tk])
            assert cost >= 0, 'Cost should be a positive value!'
            spent += cost

            A.append(tk)

            # If the k-th test's cost is over the given budget, or we're out of tests, we exit the loop
            if spent > budget or not tests:
                break

    # If the k-th test covers more pairs than all the others, return the k-th test
    if f(tests[k]) >= f([a for a in A if a != tests[k]]):
        return [tests[k]]

    # Otherwise, return all the test before the k-th
    return tests[:k - 1]
