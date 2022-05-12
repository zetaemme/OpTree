from enum import Enum


class NodeType(Enum):
    TestNode = 0
    LeafNode = 1


# FIXME: Reimplementare estendendo Node di treelib
class Node:
    """A node of the decision tree"""

    def __init__(self, label: str, node_type: NodeType) -> None:
        """Constructor of the Node Abstract Base Class"""
        self.label = label
        self.node_type = node_type
