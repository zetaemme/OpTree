import abc
from abc import ABC
from dataclasses import dataclass
from typing import Iterator, Type, Union

from test import Test


@dataclass
class Node(ABC):
    """An Abstract Base Class for the Node class"""
    label: str
    r_child: Union['TestNode', 'LeafNode']
    l_child: Union['TestNode', 'LeafNode']

    @abc.abstractmethod
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


@dataclass
class DecTree:
    """Represents a Decision Tree"""
    root: Type['Node']

    def __iter__(self) -> Iterator[Node]:
        self.current = self.root
        return self

    def __next__(self) -> Node:
        curr = self.current
        self.current = self.root.l_child if self.root.outcome else self.root.r_child

        if isinstance(curr, TestNode):
            return curr
        else:
            raise StopIteration
