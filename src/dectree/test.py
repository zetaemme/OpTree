from dataclasses import dataclass
from enum import Enum, unique
from numbers import Number, Integral, Rational
from typing import Sequence, Type, TypeVar, Union

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
    """Represents a generic test for the Decision Tree"""
    lhs: T1
    test_type: TestType
    rhs: T2
    __outcome: bool = None

    def __init__(self, *args: Union[str, Sequence[T1, str, T2]]):
        """Test class constructor"""
        if len(args) > 3:
            raise ValueError('Too many arguments!')

        if len(args) == 1:
            assert isinstance(args[0], str), 'A string should be provided as param!'
            splitted = args[0].split()

            self.test_type = TestType[splitted[1]]

            if isinstance(splitted[0], Integral):
                self.lhs = int(splitted[0])
            elif isinstance(splitted[0], Rational):
                self.lhs = float(splitted[0])
            else:
                self.lhs = splitted[0]

            if isinstance(splitted[2], Integral):
                self.rhs = int(splitted[2])
            elif isinstance(splitted[2], Rational):
                self.rhs = float(splitted[2])
            else:
                self.rhs = splitted[2]

        if len(args) == 3:
            self.lhs = args[0]
            self.test_type = TestType[args[1]]
            self.rhs = args[2]

    def __generate_outcome(self, is_direct: bool = True) -> bool:
        """Generates the outcome of a test"""
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
        """Wraps __generate_outcome() to return the corresponding boolean value"""
        if self.__outcome is not None:
            return self.__outcome

        return self.__generate_outcome(is_direct=False)

    def __str__(self) -> str:
        """Returns the string corresponding to the test"""
        return f'{self.lhs} {self.test_type} {self.rhs}'
