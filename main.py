import logging
from argparse import ArgumentParser, BooleanOptionalAction
from os.path import dirname
from pathlib import Path
from pickle import HIGHEST_PROTOCOL, Unpickler, dump
from timeit import timeit
from typing import Optional

import src
from src.dataset import Dataset
from src.decision_tree import build_decision_tree
from src.tree import Tree
from src.types import PicklePairs, PickleSeparation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - OpSion @ {%(module)s.py -> %(funcName)s} - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def benchmark_main() -> None:
    path = Path("data/test.csv")
    dataset = Dataset(path)
    build_decision_tree(dataset, src.TESTS, src.COSTS)


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
    decision_tree = Tree()
    decision_tree, _ = build_decision_tree(dataset, src.TESTS, src.COSTS, decision_tree)

    logger.info("Done!")
    decision_tree.print()

    with open(f"model/decision_tree_{name}.pkl", "wb") as obj_file:
        dump(decision_tree, obj_file, HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = ArgumentParser(prog="main.py", description="Builds (log-)optimal decision trees")
    parser.add_argument("-b", "--benchmark", action=BooleanOptionalAction)
    parser.add_argument("-f", "--filename", type=str, help="The CSV file containing the dataset")

    args = parser.parse_args()

    if args.benchmark:
        logging.disable(logging.INFO)
        time = timeit(stmt="benchmark_main()", globals=globals(), number=1000)
        print(f"Time: {time / 1000} seconds")
    else:
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
