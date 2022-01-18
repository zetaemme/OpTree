from typing import Callable

from pandas import DataFrame

from dectree.test import Test


# FIXME: Modificare con una funzione di costo accettabile
def calculate_cost(test: Test) -> int:
    return 1


# TODO: Implementare come da paper
def find_budget(objects: DataFrame, tests: list[Test], classes: list[str], cost_fn: Callable[[Test], int]):
    pass
