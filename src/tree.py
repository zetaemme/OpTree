from dataclasses import dataclass, field
from typing import Self


@dataclass
class Tree:
    """
    The decision tree. Implemented as in the following grammar:
        TREE -> epsilon | "label" TREE*
    """

    label: str | None
    children: list[Self] | None = field(default=None)

    def add_child(self, child: Self) -> None:
        """Adds the given node as a child of the labelled node, if given as parameter.
        If parent_label is not passed as parameter, the node will be added as child of root.

        Args:
            node (Node): The node to be added.
            parent_label (str, optional): The label of the given node's parent. Defaults to None.
        """
        if self.children is not None:
            self.children.append(child)
        else:
            self.children = [child]

    def set_label(self, label: str) -> None:
        """Setter for the label field

        Args:
            label (str): The root label
        """
        self.label = label
