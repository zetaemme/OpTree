from dataclasses import dataclass
from enum import Enum, unique
from numbers import Number
from typing import TypeVar

T1 = TypeVar('T1')
T2 = TypeVar('T2')


@unique
class Type(Enum):
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
    test_type: Type
    rhs: T2
    __outcome: bool = None

    def __init__(self, lhs: T1, test_type: str, rhs: T2) -> None:
        self.lhs = lhs
        self.test_type = Type[test_type]
        self.rhs = rhs

    def __generate_outcome(self, is_direct=True) -> bool:
        if not is_direct:
            match self.test_type:
                case Type.LESS_THAN:
                    assert isinstance(self.lhs, Number), 'lhs should be numeric in order to execute \'<\' comparison'
                    assert isinstance(self.rhs, Number), 'rhs should be numeric in order to execute \'<\' comparison'

                    return self.lhs < self.rhs

                case Type.GREATER_THAN:
                    assert isinstance(self.lhs, Number), 'lhs should be numeric in order to execute \'>\' comparison'
                    assert isinstance(self.rhs, Number), 'rhs should be numeric in order to execute \'>\' comparison'

                    return self.lhs > self.rhs

                case Type.EQUAL:
                    return self.lhs == self.rhs

                case Type.NOT_EQUAL:
                    return self.lhs != self.rhs

                case Type.LESS_OR_EQUAL:
                    assert isinstance(self.lhs, Number), 'lhs should be numeric in order to execute \'<=\' comparison'
                    assert isinstance(self.rhs, Number), 'rhs should be numeric in order to execute \'<=\' comparison'

                    return self.lhs <= self.rhs

                case Type.GREATER_OR_EQUAL:
                    assert isinstance(self.lhs, Number), 'lhs should be numeric in order to execute \'>=\' comparison'
                    assert isinstance(self.rhs, Number), 'rhs should be numeric in order to execute \'>=\' comparison'

                    return self.lhs >= self.rhs

    @property
    def outcome(self) -> bool:
        if self.__outcome is not None:
            return self.__outcome

        return self.__generate_outcome(is_direct=False)
