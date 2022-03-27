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

    @classmethod
    def of(cls, operator: str) -> 'TestType':
        """Converts a given operator into the corrct Enum value

        Args:
            operator (str): A string representing the operator we want to convert to

        Returns:
            TestType: The Enum value corresponding to the operator string
        """
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
    """Represents a generic test for the Decision Tree

    Attributes:
        lhs_label (str): The dataset column corresponding to the left-hand side of the test
        test_type: (TestType): The Enum value corresponding to the binary operator of the test
        rhs (Number): The value at the right-hand side of the test
        outcomes (list): The list of all possible outcomes for a test
    """
    lhs_label: str
    test_type: TestType
    rhs: Number
    outcomes: list

    def __init__(self, lhs_label: str, test_type: str, rhs: Number, outcomes: list) -> None:
        """Test class constructor

        Args:
            lhs_label (str): The dataset column corresponding to the left-hand side of the test
            test_type: (TestType): The Enum value corresponding to the binary operator of the test
            rhs (Number): The value at the right-hand side of the test
            outcomes (list): The list of all possible outcomes for a test
        """
        self.lhs_label = lhs_label
        self.test_type = TestType.of(test_type)
        self.rhs = rhs
        self.outcomes = outcomes

    def __str__(self) -> str:
        """Returns the string corresponding to the test"""
        return f'{self.lhs_label} {self.test_type} {self.rhs}'

    def evaluate_dataset_for_class(self, dataset: DataFrame, class_index: int) -> list[Series]:
        """
        Evaluates this test for all the rows in the dataset.
        Returns all the rows for which the outcome is class_index

        Args:
            dataset (DataFrame): A pandas DataFrame containing all the object we want to classify
            class_index (int): The ariety of the outcome's class

        Returns:
            list[Series]: All the rows for which the outcome of this test is class_index (as ariety)
        """
        # NOTE: Doing this assignment avoids the case in which a Generator is returned instead of a list
        result = [row for row in dataset.rows if self.outcome(row[self.lhs_label]) == class_index]
        return result

    def outcome(self, lhs_value: Number) -> int:
        """Computes the outcome of a test

        Args:
            lhs_value (Number): The value corresponding to lhs_label in the dataset

        Returns:
            int: The outcome of the test (as class ariety)
        """
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
