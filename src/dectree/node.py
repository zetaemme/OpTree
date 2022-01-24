from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union

from src.dectree.test import Test


@dataclass
class Node(ABC):
    """An Abstract Base Class for the Node class"""
    label: str
    r_child: Union['TestNode', 'LeafNode'] = field(init=True, default=None)
    l_child: Union['TestNode', 'LeafNode'] = field(init=True, default=None)

    @abstractmethod
    def outcome(self) -> bool: pass


@dataclass
class TestNode(Node):
    """Concretization of the Node class. Represents an intermediate Node"""
    __test: Test = field(init=False)

    def __post_init__(self) -> None:
        assert self.l_child is not None and self.r_child is not None, 'Tests must have children!'
        self.__test = Test(self.label)

    @property
    def outcome(self) -> bool:
        return self.__test.outcome


@dataclass
class LeafNode(Node):
    """Concretization of the Node class. Represents a leaf Node"""

    def __post_init__(self) -> None:
        assert self.l_child is None and self.r_child is None, 'Leafs shouldn\'t have any child!'

    def outcome(self) -> bool: return NotImplemented
