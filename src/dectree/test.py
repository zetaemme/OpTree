from dataclasses import dataclass
from enum import Enum, unique
from numbers import Number
from typing import TypeVar

T1 = TypeVar('T1')
T2 = TypeVar('T2')


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
    lhs: T1
    test_type: TestType
    rhs: T2

    def __init__(self, lhs: T1, test_type: str, rhs: T2):
        """Test class constructor"""
        self.lhs = lhs
        self.test_type = TestType[test_type]
        self.rhs = rhs

    @property
    def outcome(self) -> bool:
        """Generates the outcome of a test"""
        match self.test_type:
            case TestType.LESS_THAN:
                assert isinstance(self.lhs, Number), 'lhs should be numeric in order to execute \'<\' comparison'
                assert isinstance(self.rhs, Number), 'rhs should be numeric in order to execute \'<\' comparison'

                return self.lhs < self.rhs

            case TestType.GREATER_THAN:
                assert isinstance(self.lhs, Number), 'lhs should be numeric in order to execute \'>\' comparison'
                assert isinstance(self.rhs, Number), 'rhs should be numeric in order to execute \'>\' comparison'

                return self.lhs > self.rhs

            case TestType.EQUAL:
                return self.lhs == self.rhs

            case TestType.NOT_EQUAL:
                return self.lhs != self.rhs

            case TestType.LESS_OR_EQUAL:
                assert isinstance(self.lhs, Number), 'lhs should be numeric in order to execute \'<=\' comparison'
                assert isinstance(self.rhs, Number), 'rhs should be numeric in order to execute \'<=\' comparison'

                return self.lhs <= self.rhs

            case TestType.GREATER_OR_EQUAL:
                assert isinstance(self.lhs, Number), 'lhs should be numeric in order to execute \'>=\' comparison'
                assert isinstance(self.rhs, Number), 'rhs should be numeric in order to execute \'>=\' comparison'

                return self.lhs >= self.rhs

    def __str__(self) -> str:
        """Returns the string corresponding to the test"""
        return f'{self.lhs} {self.test_type} {self.rhs}'
