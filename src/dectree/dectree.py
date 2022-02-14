from dataclasses import dataclass
from typing import Iterator, Type

from node import Node

NodeType: Type['Node']


@dataclass
class DecTree:
    """Represents a Decision Tree"""
    root: NodeType

    def __iter__(self) -> Iterator[Node]:
        """Creates an Iterator, which iterates over the nodes of a Decision Tree"""
        self.current = self.root
        return self

    def __next__(self) -> Node:
        """Gets the Iterator the next node of the Decision Tree"""
        # TODO: Scegliere come deve essere tornato un nodo
        pass
