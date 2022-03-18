import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from numbers import Number
from typing import Optional, Sequence, Union

from src.dectree.test import Test
from src.utils import extract


@dataclass
class Node(ABC):
    """An Abstract Base Class for the Node class"""
    label: str
    children: list[Union['TestNode', 'LeafNode']] = field(default_factory=list)
    depth: int = field(init=False)
    parent: 'TestNode' = None

    @abstractmethod
    def add_children(self,
                     children: Union[Union['TestNode', 'LeafNode'], Sequence[Union['TestNode', 'LeafNode']]]) -> None:
        pass

    @abstractmethod
    def outcome(self, lhs_value: Optional[Number]) -> Union[int, str]: pass


@dataclass
class TestNode(Node):
    """Concretization of the Node class. Represents an intermediate Node"""
    __test: Test = field(init=False)

    def __post_init__(self) -> None:
        self.__test = extract.test_structure(self.label)
        # Removes the outcomes from the node label after initializing the test using a regex match
        self.label = re.match('^((?:\S+\s+){2}\S+).*', self.label).group(1)
        self.depth = self.parent.depth + 1 if self.children else 0

    def add_children(self,
                     children: Union[Union['TestNode', 'LeafNode'], Sequence[Union['TestNode', 'LeafNode']]]) -> None:
        if isinstance(children, Sequence):
            for child in children:
                child.parent = self
            else:
                children.parent = self

        self.children.append(children)

    def outcome(self, lhs_value: Number) -> int:
        return self.__test.outcome(lhs_value)


@dataclass
class LeafNode(Node):
    """Concretization of the Node class. Represents a leaf Node"""

    def __post_init__(self) -> None:
        assert not self.children, 'LeafNodes shouldn\'t have any child!'
        self.depth = self.parent.depth + 1 if self.children else 0

    def outcome(self, lhs_value: Optional[Number]) -> str:
        """Returns the string associated with the class ot the leaf"""
        return self.label

    def add_children(self,
                     children: Union[Union['TestNode', 'LeafNode'], Sequence[Union['TestNode', 'LeafNode']]]) -> None:
        raise RuntimeError('Cannot add children to leaf node!')
