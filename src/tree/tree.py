from dataclasses import dataclass, field

from src.tree.node import Node


@dataclass
class Tree:
    """The decision tree."""

    root: Node = field(init=False)
    intermediate_nodes: list[Node] = field(default_factory=list)

    def add_node(self, node: Node, *, parent_label=None) -> None:
        """Adds the given node as a child of the labelled node, if given as parameter.
        If parent_label is not passed as parameter, the node will be added as child of root.

        Args:
            node (Node): The node to be added.
            parent_label (str, optional): The label of the given node's parent. Defaults to None.
        """
        if parent_label is not None:
            target_node = self.find_child_labelled(parent_label)
            node.set_parent(target_node)
            target_node.add_child(node)
            self.intermediate_nodes.append(target_node)
        else:
            node.set_parent(self.root)
            self.root.add_child(node)

    def find_child_labelled(self, label: str) -> Node:
        """Finds the intermediate node labelled with the given label"""
        child = [node for node in self.intermediate_nodes if node.label == label]
        return child[0]
