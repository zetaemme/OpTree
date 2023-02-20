from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import combinations
from json import dump
from os.path import dirname

import pandas as pd


@dataclass(init=False)
class Pairs:
    """A tuple of items having different classes"""

    pairs_list: list[tuple[int, int]]

    def __init__(self, dataset: pd.DataFrame) -> None:
        if dataset.shape[0] == 1:
            self.pairs_list = []
            return

        self.pairs_list = list(
            filter(
                lambda pair: dataset["Class"][pair[0]] != dataset["Class"][pair[1]],
                combinations(dataset.index, 2)
            )
        )


def main(dataset_path: str) -> None:
    dataset = pd.read_csv(dirname(__file__) + "/" + dataset_path)
    pairs = Pairs(dataset)

    data = {"pairs": pairs.pairs_list}

    dataset_name = dataset_path.replace('data/', '').replace('.csv', '')
    with open(dirname(__file__) + f"data/pairs/{dataset_name}_pairs.json", "w") as f:
        dump(data, f)


if __name__ == '__main__':
    parser = ArgumentParser(prog="main.py", description="Builds (log-)optimal decision trees")
    parser.add_argument("-f", "--filename", type=str, help="The CSV file containing the dataset")

    args = parser.parse_args()

    main(args.filename)
