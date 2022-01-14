from typing import Callable, NewType

from pandas import DataFrame

from test import Test

uint = NewType('uint', int)


def adapted_greedy(
        objects: DataFrame,
        tests: list[Test],
        f: Callable,
        cost_f: Callable[[Test], uint],
        bound: uint
) -> list[Test]:
    """Implementation of the Adapted-Greedy heuristic"""
    assert bound >= 0, 'Bound should be a positive integer!'

    spent = 0
    A = []
    k = 0

    # Remove from T all tests with cost larger than B
    tests = [t for t in tests if cost_f(t) <= bound]

    if not tests:
        while True:
            k += 1

            # TODO: Implement 'Let tests[k] be...'

            tests.pop(k)
            spent += cost_f(tests[k])
            A.append(tests[k])

            if spent > bound or not tests:
                break

    if f(tests[k]) >= f([a for a in A if a != tests[k]]):
        return [tests[k]]

    return tests[:k - 1]
