from abc import ABC, abstractmethod
from typing import Sequence, Union


class Node(ABC):
    """An Abstract Base Class representing a Decision Tree Node"""

    def __init__(
            self,
            label: str,
            children: list[Union['TestNode', 'LeafNode']] = None,
            *, parent: 'Node' = None
    ) -> None:
        """Constructor of the Node Abstract Base Class"""
        if children is None:
            children = []

        self.label = label
        self.children = children
        self.parent = parent

        self.depth = self._calculate_depth(_user_call=False)

    @abstractmethod
    def add_children(self,
                     children: Union[Union['TestNode', 'LeafNode'], Sequence[Union['TestNode', 'LeafNode']]]) -> None:
        pass

    def _calculate_depth(self, *, _user_call=True):
        if _user_call:
            raise RuntimeError('Cannot directly invoke the \'_calculate_depth()\' method!')

        return self.parent.depth + 1 if self.parent else 0


class TestNode(Node):
    """Concretization of the Node class. Represents an intermediate Node"""

    def add_children(self,
                     children: Union[Union['TestNode', 'LeafNode'], Sequence[Union['TestNode', 'LeafNode']]]) -> None:
        """Adds children to this node

        Parameters
        ----------
        children: A single child or the list of children to add as children of this node
        """
        if isinstance(children, Sequence):
            for child in children:
                child.parent = self
            else:
                children.parent = self

        self.children.append(children)


class LeafNode(Node):
    """Concretization of the Node class. Represents a leaf Node"""

    def add_children(self,
                     children: Union[Union['TestNode', 'LeafNode'], Sequence[Union['TestNode', 'LeafNode']]]) -> None:
        """Adds children to this node

        Parameters
        ----------
        children: A single child or the list of children to add as children of this node

        Raises
        ------
        RuntimeError: A child cannot be added to a Leaf node, since this type of node represents a leaf of the
                      Decision Tree
        """
        raise RuntimeError('Cannot add children to leaf node!')
