from dataclasses import dataclass, field
from typing import Any, Iterable, Hashable, Optional, Sequence

from pandas import DataFrame, Series


@dataclass
class Dataset:
    """Interface that wraps a Pandas DataFrame, removing a functional dependency from Pandas"""
    columns: Optional[Sequence[str]]
    data: Sequence[Any]
    classes: set[str] = field(init=False)
    data_frame: DataFrame = field(init=False)
    rows: list = field(init=False)

    def __post_init__(self) -> None:
        if not self.columns:
            self.columns = [str(i) for i in range(len(self.data[0]))]

        self.data_frame = DataFrame(data=self.data, columns=self.columns)
        self.classes = {class_label for class_label in self.data_frame[['class']]}
        self.rows = [row for row in self.iterate_rows()]

    def __repr__(self) -> str:
        return self.data_frame.__repr__()

    def as_data_frame(self) -> DataFrame:
        return self.data_frame

    def iterate_rows(self) -> Iterable[tuple[Hashable, Series]]:
        return self.data_frame.iterrows()

    def get_column(self, column_name: str) -> list[Any]:
        return list(self.data_frame[column_name])

    def slice_from(self, index: Hashable) -> 'Dataset':
        return Dataset(self.data_frame[index:], self.columns)
