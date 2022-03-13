from dataclasses import dataclass
from enum import Enum, unique
from numbers import Number

from src.data.dataset import Dataset


@unique
class TestType(Enum):
    """Represents the various types of tests"""
    LESS_THAN = '<'
    GREATER_THAN = '>'
    EQUAL = '=='
    NOT_EQUAL = '!='
    LESS_OR_EQUAL = '<='
    GREATER_OR_EQUAL = '>='

    @classmethod
    def of(cls, operator: str) -> 'TestType':
        """Converts a given operator into the corrct Enum value"""
        mapping = {
            '<': TestType.LESS_THAN,
            '>': TestType.GREATER_THAN,
            '==': TestType.EQUAL,
            '!=': TestType.NOT_EQUAL,
            '<=': TestType.LESS_OR_EQUAL,
            '>=': TestType.GREATER_OR_EQUAL
        }

        return mapping[operator]


@dataclass(init=False)
class Test:
    """Represents a generic test for the Decision Tree"""
    lhs_label: str
    test_type: TestType
    rhs: Number
    outcomes: list

    def __init__(self, lhs_label: str, test_type: str, rhs: Number, outcomes: list):
        """Test class constructor"""
        self.lhs_label = lhs_label
        self.test_type = TestType.of(test_type)
        self.rhs = rhs
        self.outcomes = outcomes

    def __str__(self) -> str:
        """Returns the string corresponding to the test"""
        return f'{self.lhs_label} {self.test_type} {self.rhs}'

    def evaluate_dataset_for_class(self, dataset: Dataset, class_index: int) -> list:
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
