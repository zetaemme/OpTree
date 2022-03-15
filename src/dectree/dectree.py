from dataclasses import dataclass, field
from typing import Sequence, Union

from src.dectree.node import LeafNode, TestNode


@dataclass
class DecTree:
    """Represents a Decision Tree"""
    root: Union[LeafNode, TestNode]
    last_added_node: Union[LeafNode, TestNode] = field(init=False)

    def __post_init__(self) -> None:
        if not self.root.children:
            self.last_added_node = self.root
        else:
            # TODO: Assign last_added_node to the last added value in the children of root
            pass

    def add_children(self, children: Union[Union[LeafNode, TestNode], Sequence[Union[LeafNode, TestNode]]]) -> None:
        """Adds a children to the last added node of this tree"""
        self.last_added_node.add_children(children)

        if isinstance(children, Sequence):
            # NOTE: Since, in this branch, we're assuming you can add multiple children to the last_added_node,
            #       we use the effectively last added child to update the last_added_node value
            self.last_added_node = children[-1]
        else:
            self.last_added_node = children

    def add_subtree(self, subtree: 'DecTree') -> None:
        """Adds the DecTree subtree as a child of the last added node"""
        subtree.root.parent = self.last_added_node
        self.add_children(subtree.root)
        self.last_added_node = subtree.last_added_node
