from argparse import ArgumentParser
from dataclasses import dataclass, field
from os.path import dirname
from pickle import HIGHEST_PROTOCOL, dump, load
from typing import Any

import pandas as pd

from src.types import PicklePairs


@dataclass(init=False)
class Separation:
    _all_features: list[str]

    S_label: dict[str, dict[Any, list[int]]] = field(default_factory=dict)
    S_star: dict[str, list[int]] = field(default_factory=dict)
    sigma: dict[str, list[int]] = field(default_factory=dict)

    kept: dict[str, list[tuple[int]]] = field(default_factory=dict)
    separated: dict[str, list[tuple[int]]] = field(default_factory=dict)

    def __init__(self, dataset: pd.DataFrame, pairs: list[tuple[int, int]]):
        self.S_label = {}
        self.S_star = {}
        self.sigma = {}
        self.kept = {}  # type: ignore
        self.separated = {}  # type: ignore

        if "Probability" in dataset.columns:
            self._all_features = dataset.columns.values[:-2]
        else:
            self._all_features = dataset.columns.values[:-1]

        for feature_idx, feature in enumerate(self._all_features):
            self.S_label[feature] = {
                value: [idx for idx, item in dataset.iterrows() if item[feature] == value]
                for value in dataset[feature].unique().tolist()
            }

            feature_pairs = {
                label: number_of_pairs_for(pairs, objects) for label, objects in self.S_label[feature].items()
            }
            max_pairs = max(feature_pairs, key=feature_pairs.get)  # type: ignore
            self.S_star[feature] = self.S_label[feature][max_pairs]

            self.sigma[feature] = [int(idx) for idx in set(dataset.index.values) - set(self.S_star[feature])]

            self.kept[feature] = list(  # type: ignore
                filter(lambda pair: all(idx in self.sigma[feature] for idx in pair), pairs)
            )

            self.separated[feature] = list(  # type: ignore
                filter(
                    lambda pair: pair[0] in self.sigma[feature]
                                 and pair[1] in self.S_star[feature]
                                 or pair[1] in self.sigma[feature]
                                 and pair[0] in self.S_star[feature],
                    pairs,
                )
            )


def number_of_pairs_for(pairs: list[tuple[int, int]], objects: list[int]) -> int:
    return len({pair for obj in objects for pair in pairs if int(obj) in pair})


def main(dataset_path: str, pairs: list[tuple[int, int]]) -> None:
    dataset = pd.read_csv(dirname(__file__) + f"/{dataset_path}")

    separation = Separation(dataset, pairs)

    dataset_name = dataset_path.replace("data/", "").replace(".csv", "")
    with open(f"./data/separation/{dataset_name}_separation.pkl", "wb") as f:
        dump(
            {
                "S_label": separation.S_label,
                "S_star": separation.S_star,
                "sigma": separation.sigma,
                "separated": separation.separated,
                "kept": separation.kept,
            },
            f,
            HIGHEST_PROTOCOL
        )


if __name__ == "__main__":
    parser = ArgumentParser(prog="compute_pairs.py", description="Computes the pairs for the procedure")
    parser.add_argument("-f", "--filename", type=str, help="The CSV file containing the dataset")
    parser.add_argument("-p", "--pairs", type=str, help="The JSON file containing the pre-computed pairs")

    args = parser.parse_args()

    with open(dirname(__file__) + f"/{args.pairs}", "rb") as f:
        pickle_pairs: PicklePairs = load(f)
        pairs = [
            tuple(pair)
            for pair in pickle_pairs["pairs"]
        ]

    main(args.filename, pairs)  # type: ignore
