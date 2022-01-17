from typing import Callable, NewType, TypeAlias

from pandas import DataFrame

from dectree.test import Test

uint = NewType('uint', int)
TestList: TypeAlias = list[Test]


def adapted_greedy(
        objects: DataFrame,
        tests: TestList,
        f: Callable,
        cost_fn: Callable[[Test], uint],
        bound: uint
) -> TestList:
    """Implementation of the Adapted-Greedy heuristic"""
    assert bound >= 0, 'Bound should be a positive integer!'

    spent = 0
    A = []
    k = 0

    # Remove from T all tests with cost larger than B
    tests = [t for t in tests if cost_fn(t) <= bound]

    if not tests:
        while True:
            k += 1

            tests.pop(k)

            cost = cost_fn(tests[k])
            assert cost >= 0, 'Cost should be a positive value!'
            spent += cost

            A.append(tests[k])

            if spent > bound or not tests:
                break

    if f(tests[k]) >= f([a for a in A if a != tests[k]]):
        return [tests[k]]

    return tests[:k - 1]
