from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import combinations
from os.path import dirname
from pickle import HIGHEST_PROTOCOL, dump

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
            filter(lambda pair: dataset["Class"][pair[0]] != dataset["Class"][pair[1]], combinations(dataset.index, 2))
        )


def compute_pairs(dataset_path: str) -> None:
    dataset = pd.read_csv(dirname(__file__) + "/" + dataset_path)
    pairs = Pairs(dataset)

    data = {"pairs": pairs.pairs_list}

    dataset_name = dataset_path.replace("data/", "").replace(".csv", "")
    with open(dirname(__file__) + f"/data/pairs/{dataset_name}_pairs.pkl", "wb") as f:
        dump(data, f, HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = ArgumentParser(prog="compute_pairs.py", description="Computes the pairs for the procedure")
    parser.add_argument("-f", "--filename", type=str, help="The CSV file containing the dataset")

    args = parser.parse_args()

    compute_pairs(args.filename)
