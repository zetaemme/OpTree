from typing import Callable, TypeAlias

from pandas import DataFrame

from dectree.test import Test

TestList: TypeAlias = list[Test]


def calculate_cost(test: Test) -> int:
    """Calculates the cost of a given test"""
    return 1


# FIXME: Dovrebbe tornare int, ma si adatta ll'attuale cost_fn
def calculate_separation_cost(tests: TestList) -> float:
    """Calculates the separation cost of a given test list"""
    # TODO: Ritorna sepcost come da paper
    pass


# TODO: Implementare come da paper
def find_budget(objects: DataFrame, tests: list[Test], classes: set[str], cost_fn: Callable[[Test], float]):
    """Implementation of the FindBudget procedure of the referenced paper"""
    pass
