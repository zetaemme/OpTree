from typing import TypeAlias

from pandas import DataFrame

from src.dectree.test import Test

TestList: TypeAlias = list[Test]


def extract_cheapest_test(tests: TestList, objects: tuple) -> Test:
    """Extracts the cheapest (separation cost) test that separates the two objects"""
    assert len(objects) == 2, "The tuple should contain only 2 items!"

    if len(tests) == 1:
        return tests[0]

    # TODO: Ritornare il test con sepcost minimo


def extract_object_class(dataset: DataFrame, index: int) -> str:
    """Extracts the class label from the item in position index of a given dataset"""
    assert index >= 0, "Index should be a positive integer"
    return dataset[index]['class']
