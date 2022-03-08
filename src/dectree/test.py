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
    outcomes: dict[bool, int]

    def __init__(self, lhs_label: str, test_type: str, rhs: Number, true_outcome: int, false_outcome: int):
        """Test class constructor"""
        self.lhs_label = lhs_label
        self.test_type = TestType[test_type]
        self.rhs = rhs

        self.outcomes[True] = true_outcome
        self.outcomes[False] = false_outcome

    def __str__(self) -> str:
        """Returns the string corresponding to the test"""
        return f'{self.lhs_label} {self.test_type} {self.rhs}'

    # FIXME: Rimuovere dipendenza strutturale da pandas con un'interfaccia
    def evaluate_dataset_for_class(self, dataset: DataFrame, class_index: int) -> list[Series]:
        """
        Evaluates this test for all the rows in the dataset.
        Returns all the rows for which the outcome is class_index
        """
        # NOTE: Doing this assignment avoids the case in which a Generator is returned instead of a list
        result = [row for row in dataset.rows if self.outcome(row[self.lhs_label]) == class_index]

        return result

    def outcome(self, lhs_value: Number) -> int:
        """Generates the outcome of a test"""
        # NOTE: Assume that each node has only a binary outcome
        match self.test_type:
            case TestType.LESS_THAN:
                return self.outcomes[lhs_value < self.rhs]

            case TestType.GREATER_THAN:
                return self.outcomes[lhs_value > self.rhs]

            case TestType.EQUAL:
                return self.outcomes[lhs_value == self.rhs]

            case TestType.NOT_EQUAL:
                return self.outcomes[lhs_value != self.rhs]

            case TestType.LESS_OR_EQUAL:
                return self.outcomes[lhs_value <= self.rhs]

            case TestType.GREATER_OR_EQUAL:
                return self.outcomes[lhs_value >= self.rhs]
