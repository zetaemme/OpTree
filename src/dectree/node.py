import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from numbers import Number
from typing import Optional, Sequence, Union


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

    def __post_init__(self) -> None:
        # Removes the outcomes from the node label after initializing the test using a regex match
        self.depth = self.parent.depth + 1 if self.children else 0

    def add_children(self,
                     children: Union[Union['TestNode', 'LeafNode'], Sequence[Union['TestNode', 'LeafNode']]]) -> None:
        """Adds children to this node

        Args:
            children (Union[Union['TestNode', 'LeafNode'], Sequence[Union['TestNode', 'LeafNode']]]):
                A single child or the list of children to add as children of this node
        """
        if isinstance(children, Sequence):
            for child in children:
                child.parent = self
            else:
                children.parent = self

        self.children.append(children)

    def outcome(self, lhs_value: Number) -> int:
        """Computes the outcome of the node, wrapping Test#outcome()

        Args:
            lhs_value (str): A string corresponding to the column of the dataset to be used ad left-hand side of the test

        Returns:
            int: The outcome of the test (as class ariety)
        """
        return self.__test.outcome(lhs_value)


@dataclass
class LeafNode(Node):
    """Concretization of the Node class. Represents a leaf Node"""
    def __post_init__(self) -> None:
        assert not self.children, 'LeafNodes shouldn\'t have any child!'
        self.depth = self.parent.depth + 1 if self.children else 0

    def outcome(self, lhs_value: Optional[Number]) -> str:
        """Returns the string associated with the class ot the leaf

        Args:
            lhs_value (str, optional): A string corresponding to the column of the dataset to be used ad left-hand side
                                       of the test

        Returns:
            int: The outcome of the test (as class ariety)
        """
        return self.label

    def add_children(self,
                     children: Union[Union['TestNode', 'LeafNode'], Sequence[Union['TestNode', 'LeafNode']]]) -> None:
        """Adds children to this node

        Args:
            children (Union[Union['TestNode', 'LeafNode'], Sequence[Union['TestNode', 'LeafNode']]]):
                A single child or the list of children to add as children of this node

        Raises:
              RuntimeError: A child cannot be added to a Leaf node, since this type of node represents a leaf of the
                            Decision Tree
        """
        raise RuntimeError('Cannot add children to leaf node!')
