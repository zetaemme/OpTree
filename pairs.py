from dataclasses import dataclass, field
from pandas import DataFrame


@dataclass
class Pairs:
    dataset: DataFrame
    number: int = field(init=False)
    pair_list: list[tuple[int]] = field(init=False)

    def __post_init__(self) -> None:
        """Inits the number field"""
        # We suppose to have a 'class' column in the dataset
        assert 'class' in self.dataset.columns, 'Dataset should contain a \'class\' column'

        # FIXME: Convert the list init to a list comprehension
        for i1, l1 in enumerate(self.dataset):
            for i2, l2 in enumerate(self.dataset[i1:]):
                if l2['class'] is not None and l1['class'] != l2['class']:
                    self.number += 1

                    pair = tuple([i1, i2])
                    if (i2, i1) not in self.pair_list:
                        self.pair_list.append(pair)
