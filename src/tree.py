from dataclasses import dataclass, field
from typing import Self


@dataclass(repr=False)
class Tree:
    """
    The decision tree. Implemented as in the following grammar:
        TREE -> epsilon | "label" TREE*
    """

    label: str | None
    children: list[Self] | None = field(default=None)
    last_added: Self = field(init=False)

    def add_child(self, child: Self) -> None:
        """Adds the given node as a child of this tree.

        Args:
            child (Self): The node to be added.
        """
        if self.children is None:
            self.children = [child]
        else:
            self.children.append(child)

        self.last_added = child

    def add_children(self, children: list[Self]) -> None:
        """Adds the given nodes as children of this tree.

        Args:
            children (list[Self]): The nodes to be added.
        """
        if self.children is None:
            self.children = children
        else:
            self.children += children

        self.last_added = children[-1]

    def set_label(self, label: str) -> None:
        """Setter for the label field

        Args:
            label (str): The root label
        """
        self.label = label
        self.last_added = self

    def __repr__(self) -> str:
        if self.children is not None:
            child_repr = [child.__repr__() for child in self.children]
        else:
            child_repr = ""

        if child_repr:
            return f"{self.label} -> [ {', '.join(child_repr)} ]"
        else:
            return self.label
