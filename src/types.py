from dataclasses import dataclass
from typing import Callable, NewType

from .dataset import Dataset
from .separation import Separation


@dataclass(repr=False, slots=True)
class Bounds:
    lower: float
    upper: float


SubmodularFunction = Callable[[Dataset, Separation, list[str]], int]

HeuristicFunction = Callable[
    [float, Dataset, Separation, SubmodularFunction], list[str]
]
