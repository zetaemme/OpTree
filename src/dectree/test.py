from dataclasses import dataclass
from enum import Enum, unique
from numbers import Number

from pandas import DataFrame, Series


@unique
class TestType(Enum):
    """Represents the various types of tests"""
    LESS_THAN = '<'
    GREATER_THAN = '>'
    EQUAL = '=='
    NOT_EQUAL = '!='
    LESS_OR_EQUAL = '<='
    GREATER_OR_EQUAL = '>='


@dataclass(init=False)
class Test:
    """Represents a generic test for the Decision Tree"""
    lhs_label: str
    test_type: TestType
    rhs: Number

    def __init__(self, lhs_label: str, test_type: str, rhs: Number):
        """Test class constructor"""
        self.lhs_label = lhs_label
        self.test_type = TestType[test_type]
        self.rhs = rhs

    def __str__(self) -> str:
        """Returns the string corresponding to the test"""
        return f'{self.lhs_label} {self.test_type} {self.rhs}'

    # FIXME: Rimuovere dipendenza strutturale da pandas con un'interfaccia
    def evaluate_dataset_for_class(self, dataset: DataFrame, class_label: int) -> list[Series]:
        """
        Evaluates this test for all the rows in the dataset.
        Returns all the rows for which the outcome is class_label
        """
        return [row for row in dataset.rows if self.outcome(row[self.lhs_label]) == class_label]

    def outcome(self, lhs_value: Number) -> int:
        """Generates the outcome of a test"""
        """
        match self.test_type:
            case TestType.LESS_THAN:
                return lhs_value < self.rhs

            case TestType.GREATER_THAN:
                return lhs_value > self.rhs

            case TestType.EQUAL:
                return lhs_value == self.rhs

            case TestType.NOT_EQUAL:
                return lhs_value != self.rhs

            case TestType.LESS_OR_EQUAL:
                return lhs_value <= self.rhs

            case TestType.GREATER_OR_EQUAL:
                return lhs_value >= self.rhs
        """
        # TODO: Implementare correttamente
        return 0
