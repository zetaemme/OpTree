from dataclasses import dataclass, field
from typing import TypeAlias

from pandas import DataFrame

TupleList: TypeAlias = list[tuple]


@dataclass
class Pairs:
    __dataset: DataFrame = field(repr=False)
    number: int = field(init=False)
    pair_list: TupleList = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        """Inits the number and pair_list fields"""
        # We suppose to have a 'class' column in the dataset
        assert 'class' in self.__dataset.columns, 'Dataset should contain a \'class\' column'

        self.pair_list = [
            (i1, i2)
            for i1, d1 in self.__dataset.iterrows()
            for i2, d2 in self.__dataset[i1:].iterrows()
            if d1['class'] != d2['class']
        ]

        self.number = len(self.pair_list)
