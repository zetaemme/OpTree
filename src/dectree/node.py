from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union

from src.dectree.test import Test
from src.utils import extract


@dataclass
class Node(ABC):
    """An Abstract Base Class for the Node class"""
    label: str = field(init=True)
    children: list[Union['TestNode', 'LeafNode']] = field(default=None)

    @abstractmethod
    def outcome(self) -> bool: pass


@dataclass
class TestNode(Node):
    """Concretization of the Node class. Represents an intermediate Node"""
    __test: Test = field(init=False)

    def __post_init__(self) -> None:
        # TODO: Aggiungere calcolo degli oggetti separati per ogni classe
        assert self.children, 'TestNodes must have children!'
        self.__test = extract.test_structure(self.label)

    @property
    def outcome(self) -> bool:
        return self.__test.outcome


@dataclass
class LeafNode(Node):
    """Concretization of the Node class. Represents a leaf Node"""

    def __post_init__(self) -> None:
        assert not self.children, 'LeafNodes shouldn\'t have any child!'

    def outcome(self) -> str:
        """Returns the string associated with the class ot the leaf"""
        return self.label
