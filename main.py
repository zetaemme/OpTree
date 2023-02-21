import logging
from argparse import ArgumentParser, BooleanOptionalAction
from json import load
from os.path import dirname
from pathlib import Path
from pickle import HIGHEST_PROTOCOL, dump
from timeit import timeit
from typing import Literal

from src.dataset import Dataset
from src.decision_tree import build_decision_tree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - OpSion @ {%(module)s.py -> %(funcName)s} - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def benchmark_main() -> None:
    path = Path("data/test.csv")
    dataset = Dataset(path)
    build_decision_tree(dataset)


def main(dataset_path: str, dataset_pairs: list[tuple[int]] | None = None) -> None:
    """Inits dataset and runs the algorithm"""
    path = Path(dirname(__file__) + "/" + dataset_path)

    if pairs is not None:
        dataset = Dataset(path, dataset_pairs)
    else:
        dataset = Dataset(path)

    decision_tree, _ = build_decision_tree(dataset)

    logger.info("Done!")
    decision_tree.print()

    dataset_name = dataset_path.replace('data/', '').replace('.csv', '')
    with open(f"model/decision_tree_{dataset_name}.obj", "wb") as obj_file:
        dump(decision_tree, obj_file, HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = ArgumentParser(prog="main.py", description="Builds (log-)optimal decision trees")
    parser.add_argument("-b", "--benchmark", action=BooleanOptionalAction)
    parser.add_argument("-f", "--filename", type=str, help="The CSV file containing the dataset")
    parser.add_argument(
        "-p",
        "--pairs",
        default=None,
        type=str,
        help="The JSON file containing the pre-computed pairs for the dataset"
    )

    args = parser.parse_args()

    if args.benchmark:
        logging.disable(logging.INFO)
        time = timeit(stmt="benchmark_main()", globals=globals(), number=1000)
        print(f"Time: {time / 1000} seconds")
    else:
        if args.pairs is not None:
            with open(dirname(__file__) + "/" + args.pairs, "r") as f:
                json_pairs: dict[Literal["pairs"], list[list[int]]] = load(f)
                pairs = [
                    tuple(pair)
                    for pair in json_pairs["pairs"]
                ]

                main(args.filename, pairs)
        else:
            main(args.filename)
