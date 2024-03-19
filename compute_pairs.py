from argparse import ArgumentParser
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool, cpu_count
from os.path import dirname
from pickle import HIGHEST_PROTOCOL, dump

import pandas as pd
import numpy as np


@dataclass(init=False)
class Pairs:
    """A tuple of items having different classes"""

    pairs_list: set[tuple[int, int]]

    def __init__(self, dataset: pd.DataFrame) -> None:
        if dataset.shape[0] == 1:
            self.pairs_list = set()
            return

        # Divide the list of indices into chunks
        chunks = np.array_split(np.sort(dataset.index), cpu_count())

        # Create a multiprocessing pool and compute the pairs in parallel
        with Pool() as pool:
            pairs_sets = pool.map(partial(self.parallel_compute_pairs, dataset=dataset), chunks)

        # Union the sets into a single set
        self.pairs_list = sorted(set().union(*pairs_sets))

    def parallel_compute_pairs(self, chunk, dataset):
        pairs = set()
        for i in chunk:
            for j in range(i + 1, len(dataset.index)):
                if dataset["Class"][i] != dataset["Class"][j]:
                    pairs.add((i, j))
        return pairs


def compute_pairs(dataset_path: str) -> None:
    dataset = pd.read_csv(dirname(__file__) + "/" + dataset_path)
    pairs = Pairs(dataset)

    data = {"pairs": list(pairs.pairs_list)}

    dataset_name = dataset_path.replace("data/datasets/csv/", "").replace(".csv", "")
    with open(dirname(__file__) + f"/data/pairs/{dataset_name}_pairs.pkl", "wb") as f:
        dump(data, f, HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = ArgumentParser(prog="compute_pairs.py", description="Computes the pairs for the procedure")
    parser.add_argument("-f", "--filename", type=str, help="The CSV file containing the dataset")

    args = parser.parse_args()

    compute_pairs(args.filename)
