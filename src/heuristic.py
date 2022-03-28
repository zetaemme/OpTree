from typing import Callable

from dectree.test import Test


def adapted_greedy(
        # FIXME: In the paper there's S as parameter, but it's never used
        tests: list[Test],
        test_costs: dict[Test, int],
        f: Callable,
        # NOTE: By computing the test cost at the beginning of the procedure, we can achieve constant lookup time
        #       extracting the cost of a certain test.
        #       This holds on the assumption that the cost of a test can't change during the execution.
        # cost_fn: Callable[[Test], int],
        budget: int
) -> list[Test]:
    """Implementation of the Adapted-Greedy heuristic

    Args:
        tests (list[Test]): A list containing all the tests
        test_costs (dict[Test, int]): A dictionary containing, for each test, the corresponding effective cost
        f (Callable): A submodular function
        cost_fn (Callable[[Test], int]): A function that calculates the effective cost of a given test
        budget (int): The maximum budget that a test list should not cross

    Returns:
        list[Test]: The maximum list of tests that can be used without crossing the budget
    """
    assert budget >= 0, 'Bound should be a positive integer!'

    spent = 0
    A = []
    k = 0

    # Remove from T all tests with cost larger than B
    tests = [test for test in tests if test_costs[test] <= budget]

    if not tests:
        while True:
            k += 1

            # Removes and returns the k-th item from the test list
            tk = tests.pop(k)

            # Calculate the cost of the k-th test and add it to 'spent'
            cost = test_costs[tk]
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
