from dataclasses import dataclass
from typing import Iterator, Type

from node import Node, TestNode


@dataclass
class DecTree:
    """Represents a Decision Tree"""
    root: Type['Node']

    def __iter__(self) -> Iterator[Node]:
        self.current = self.root
        return self

    def __next__(self) -> Node:
        curr = self.current
        self.current = self.root.l_child if self.root.outcome else self.root.r_child

        if isinstance(curr, TestNode):
            return curr
        else:
            raise StopIteration
