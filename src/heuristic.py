from typing import Callable, TypeAlias

from dectree.test import Test

TestList: TypeAlias = list[Test]


def adapted_greedy(
        # FIXME: Sul paper qui c'è S, che però viene utilizzato solo da f
        tests: TestList,
        f: Callable,
        cost_fn: Callable[[Test], int],
        budget: int
) -> TestList:
    """Implementation of the Adapted-Greedy heuristic"""
    assert budget >= 0, 'Bound should be a positive integer!'

    spent = 0
    A = []
    k = 0

    # Remove from T all tests with cost larger than B
    tests = [t for t in tests if cost_fn(t) <= budget]

    if not tests:
        while True:
            k += 1

            # Removes and returns the k-th item from the test list
            tk = tests.pop(k)

            # Calculate the cost of the k-th test and add it to 'spent'
            cost = cost_fn(tk)
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
