from typing import Callable

from pandas import DataFrame

from dectree.test import Test


# FIXME: Modificare con una funzione di costo accettabile.
#        Per ora è stata utilizzata la variante con costo 1/2 del paper
def calculate_cost(test: Test) -> float:
    """Calculates the cost of a given test"""
    return 0.5


# TODO: Implementare come da paper
def find_budget(objects: DataFrame, tests: list[Test], classes: set[str], cost_fn: Callable[[Test], float]):
    """Implementation of the FindBudget procedure of the referenced paper"""
    pass
