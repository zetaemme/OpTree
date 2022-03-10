from dataclasses import dataclass
from typing import Union

from src.dectree.node import LeafNode, TestNode


@dataclass
class DecTree:
    """Represents a Decision Tree"""
    root: Union[LeafNode, TestNode]
