from dataclasses import dataclass, field

from src.tree.node import Node


@dataclass
class Tree:
    root: Node = field(init=False)
    children: list[Node] = field(init=False, default_factory=list)
