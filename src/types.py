from dataclasses import dataclass
from typing import Callable, Literal

from src.dataset import Dataset


@dataclass(repr=False, slots=True)
class Bounds:
    """Defines the bounds for the binary search"""

    lower: float
    upper: float


SubmodularFunction = Callable[[Dataset, list[str]], int]

HeuristicFunction = Callable[
    [float, Dataset, SubmodularFunction], list[str]
]

Nodes = list[dict[Literal["id", "name"], str]]
Edges = list[dict[Literal["source", "target", "label"], str]]
