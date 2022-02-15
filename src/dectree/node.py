from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from numbers import Number
from typing import Optional, Union

from src.dectree.test import Test
from src.utils import extract


@dataclass
class Node(ABC):
    """An Abstract Base Class for the Node class"""
    label: str = field(init=True)
    children: list[Union['TestNode', 'LeafNode']] = field(default=None)

    @abstractmethod
    def outcome(self, lhs_value: Optional[Number]) -> Union[int, str]: pass


@dataclass
class TestNode(Node):
    """Concretization of the Node class. Represents an intermediate Node"""
    __test: Test = field(init=False)

    def __post_init__(self) -> None:
        assert 0 < len(self.children) <= 2, 'TestNodes must have 1 or 2 children!'
        self.__test = extract.test_structure(self.label)
        # Removes the outcomes from the node label after initializing the test
        self.label = self.label[:-4]

    def outcome(self, lhs_value: Number) -> int:
        return self.__test.outcome(lhs_value)


@dataclass
class LeafNode(Node):
    """Concretization of the Node class. Represents a leaf Node"""

    def __post_init__(self) -> None:
        assert not self.children, 'LeafNodes shouldn\'t have any child!'

    def outcome(self, lhs_value: Optional[Number]) -> str:
        """Returns the string associated with the class ot the leaf"""
        return self.label
