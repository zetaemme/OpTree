from dataclasses import dataclass
from enum import Enum, unique
from numbers import Number, Integral, Rational
from typing import Callable, Type, TypeVar

T1 = TypeVar('T1')
T2 = TypeVar('T2')

Numeric: Type[Number]


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
    """Represents a generic test for the DT"""
    lhs: T1
    test_type: TestType
    rhs: T2
    __outcome: bool = None

    def __init__(self, lhs: T1, test_type: str, rhs: T2) -> None:
        self.lhs = lhs
        self.test_type = TestType[test_type]
        self.rhs = rhs

    def __generate_outcome(self, is_direct=True) -> bool:
        if not is_direct:
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
        else:
            raise Exception('Can\'t directly invoke the __generate_outcome method.')

    @property
    def outcome(self) -> bool:
        if self.__outcome is not None:
            return self.__outcome

        return self.__generate_outcome(is_direct=False)
