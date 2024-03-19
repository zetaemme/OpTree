from argparse import ArgumentParser
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Pool, cpu_count
from pickle import HIGHEST_PROTOCOL, dump, load

import pandas as pd

from src.types import PicklePairs


@dataclass(init=False)
class Separation:
    _all_features: list[str]

    S_label: dict[str, dict[str, list[int]]] = field(default_factory=dict)
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

        with Pool(cpu_count()) as pool:
            results = pool.map(partial(self.parallel_compute_feature, dataset=dataset, pairs=pairs), self._all_features)

        for feature, S_label, S_star, sigma, kept, separated in results:
            self.S_label[feature] = S_label
            self.S_star[feature] = S_star
            self.sigma[feature] = sigma
            self.kept[feature] = kept
            self.separated[feature] = separated

    def parallel_compute_feature(self, feature, dataset, pairs):
        S_label = {
            str(value): [
                int(idx)
                for idx, item in dataset.iterrows()
                if item[feature] == value
            ]
            for value in dataset[feature].unique()
        }

        feature_pairs = {
            label: number_of_pairs_for(pairs, objects)
            for label, objects in S_label.items()
        }
        max_pairs = max(feature_pairs, key=feature_pairs.get)
        S_star = S_label[max_pairs]

        sigma = [int(idx) for idx in set(dataset.index.values) - set(S_star)]

        kept = list(
            filter(lambda pair: all(idx in sigma for idx in pair), pairs)
        )

        separated = list(
            filter(
                lambda pair: pair[0] in sigma
                             and pair[1] in S_star
                             or pair[1] in sigma
                             and pair[0] in S_star,
                pairs,
            )
        )

        return feature, S_label, S_star, sigma, kept, separated


def number_of_pairs_for(pairs: list[tuple[int, int]], objects: list[int]) -> int:
    return len({
        pair
        for obj in objects
        for pair in pairs
        if obj in pair and pair[0] in objects and pair[1] in objects
    })


def compute_separation(dataset_path: str, pairs_path: str) -> None:
    dataset = pd.read_csv(dataset_path)

    with open(pairs_path, "rb") as f:
        pickle_pairs: PicklePairs = load(f)
        pairs = [
            tuple(pair)
            for pair in pickle_pairs["pairs"]
        ]

    separation = Separation(dataset, pairs)  # type: ignore

    dataset_name = dataset_path.replace("data/datasets/csv/", "").replace(".csv", "")
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

    compute_separation(args.filename, args.pairs)  # type: ignore
