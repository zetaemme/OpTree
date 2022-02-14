from dataclasses import dataclass
from enum import Enum, unique
from numbers import Number


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
    lhs: Number
    test_type: TestType
    rhs: Number

    def __init__(self, lhs: Number, test_type: str, rhs: Number):
        """Test class constructor"""
        self.lhs = lhs
        self.test_type = TestType[test_type]
        self.rhs = rhs

    @property
    def outcome(self) -> bool:
        """Generates the outcome of a test"""
        match self.test_type:
            case TestType.LESS_THAN:
                return self.lhs < self.rhs

            case TestType.GREATER_THAN:
                return self.lhs > self.rhs

            case TestType.EQUAL:
                return self.lhs == self.rhs

            case TestType.NOT_EQUAL:
                return self.lhs != self.rhs

            case TestType.LESS_OR_EQUAL:
                return self.lhs <= self.rhs

            case TestType.GREATER_OR_EQUAL:
                return self.lhs >= self.rhs

    def __str__(self) -> str:
        """Returns the string corresponding to the test"""
        return f'{self.lhs} {self.test_type} {self.rhs}'
