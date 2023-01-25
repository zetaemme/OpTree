from dataclasses import dataclass
from typing import Callable

from src.dataset import Dataset
from src.separation import Separation


@dataclass(repr=False, slots=True)
class Bounds:
    """Defines the bounds for the binary search"""

    lower: float
    upper: float


SubmodularFunction = Callable[[Dataset, Separation, list[str]], int]

HeuristicFunction = Callable[
    [float, Dataset, Separation, SubmodularFunction], list[str]
]
