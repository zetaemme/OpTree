from dataclasses import dataclass, field
from typing import Self


@dataclass
class Node:
    """A node in the decision tree."""

    label: str
    parent: Self | None = field(default=None)
    children: list[Self] = field(init=False, default_factory=list)

    def add_child(self, node: Self) -> None:
        """Adds the given node as child of this node.

        Args:
            node (Self): The node to be added as child.
        """
        self.children.append(node)

    def set_parent(self, parent: Self) -> None:
        """Updates the parent of this node.

        Args:
            parent (Self): The node to be set as parent.
        """
        self.parent = parent
