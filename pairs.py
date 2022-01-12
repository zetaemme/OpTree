from dataclasses import dataclass

from pandas import DataFrame


@dataclass
class Pairs:
    dataset: DataFrame

    def __post_init__(self):
        pass
