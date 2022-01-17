from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

from test import Test


@dataclass
class Node(ABC):
    """An Abstract Base Class for the Node class"""
    label: str
    r_child: Union['TestNode', 'LeafNode']
    l_child: Union['TestNode', 'LeafNode']

    @abstractmethod
    def outcome(self) -> bool: pass


@dataclass
class TestNode(Node):
    """Concretization of the Node class. Represents an intermediate Node"""
    __test: Test

    @property
    def outcome(self) -> bool:
        return self.__test.outcome


@dataclass
class LeafNode(Node):
    """Concretization of the Node class. Represents a leaf Node"""

    def __post_init__(self) -> None:
        assert self.l_child is None and self.r_child is None, 'Leafs shouldn\'t have any child!'

    def outcome(self) -> bool: return NotImplemented
