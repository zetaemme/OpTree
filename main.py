import logging
from argparse import ArgumentParser
from os.path import dirname
from pathlib import Path
from pickle import Unpickler
from typing import Optional

import src
from src.dataset import Dataset
from src.decision_tree import DecisionTree
from src.types import PicklePairs, PickleSeparation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - OpSion @ {%(module)s.py -> %(funcName)s} - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("decision_tree")


def main(
        dataset_path: str,
        dataset_pairs: Optional[PicklePairs],
        dataset_separation: Optional[PickleSeparation],
        name: str
) -> None:
    """Inits dataset and runs the algorithm"""
    path = Path(dirname(__file__) + f"/{dataset_path}")

    if dataset_pairs is not None and dataset_separation is not None:
        dataset = Dataset(path, dataset_pairs, dataset_separation)
    elif dataset_separation is None:
        dataset = Dataset(path, dataset_pairs, None)
    else:
        dataset = Dataset(path)

    # NOTE: 31/03/2023 - Since it happens to have structurally equal objects with different class label, we remove the
    #                    most represented one. Doing so, we assure the purity of the leaves while keeping intact the
    #                    variance of data.
    dataset.drop_equal_objects_with_different_class()

    src.TESTS = dataset.features
    src.COSTS = dataset.costs

    decision_tree = DecisionTree()
    model = decision_tree.fit(dataset, src.TESTS, src.COSTS, name)

    model.print()


if __name__ == "__main__":
    parser = ArgumentParser(prog="main.py", description="Builds (log-)optimal decision trees")
    parser.add_argument("-f", "--filename", type=str, help="The CSV file containing the dataset")

    args = parser.parse_args()

    dataset_name = args.filename.replace('data/', '').replace('.csv', '')
    pairs = None
    separation = None

    if Path(dirname(__file__) + f"/data/pairs/{dataset_name}_pairs.pkl").is_file():
        with open(dirname(__file__) + f"/data/pairs/{dataset_name}_pairs.pkl", "rb") as pairs_f:
            logger.info("Loading pairs from Pickle file")
            unpickler = Unpickler(pairs_f)
            pickle_pairs: PicklePairs = unpickler.load()
            pairs = pickle_pairs["pairs"]

    if Path(dirname(__file__) + f"/data/separation/{dataset_name}_separation.pkl").is_file():
        with open(dirname(__file__) + f"/data/separation/{dataset_name}_separation.pkl", "rb") as separation_f:
            logger.info("Loading separation from Pickle file")
            unpickler = Unpickler(separation_f)
            separation: PickleSeparation = unpickler.load()

    main(args.filename, pairs, separation, dataset_name)
