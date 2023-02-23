import logging
from argparse import ArgumentParser, BooleanOptionalAction
from json import load
from os.path import dirname
from pathlib import Path
from pickle import HIGHEST_PROTOCOL, dump
from timeit import timeit

from src.dataset import Dataset
from src.decision_tree import build_decision_tree
from src.types import PairsJson, SeparationJson
from src.utils import parse

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


def main(dataset_path: str, dataset_pairs: PairsJson | None, dataset_separation: SeparationJson | None) -> None:
    """Inits dataset and runs the algorithm"""
    path = Path(dirname(__file__) + f"/{dataset_path}")

    if dataset_pairs is not None and dataset_separation is not None:
        dataset = Dataset(path, dataset_pairs, dataset_separation)
    elif dataset_separation is None:
        dataset = Dataset(path, dataset_pairs, None)
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

    args = parser.parse_args()

    if args.benchmark:
        logging.disable(logging.INFO)
        time = timeit(stmt="benchmark_main()", globals=globals(), number=1000)
        print(f"Time: {time / 1000} seconds")
    else:
        dataset_name = args.filename.replace('data/', '').replace('.csv', '')
        pairs = None
        separation = None

        if Path(dirname(__file__) + f"/data/pairs/{dataset_name}_pairs.json").is_file():
            with open(dirname(__file__) + f"/data/pairs/{dataset_name}_pairs.json", "r") as f:
                logger.info("Loading pairs from JSON file")
                json_pairs: PairsJson = load(f)
                pairs = [
                    tuple(pair)
                    for pair in json_pairs["pairs"]
                ]

        if Path(dirname(__file__) + f"/data/separation/{dataset_name}_separation.json").is_file():
            with open(dirname(__file__) + f"/data/separation/{dataset_name}_separation.json", "r") as f:
                logger.info("Loading separation from JSON file")
                json_separation: SeparationJson = load(f)
                separation = {
                    "S_label": {
                        key: parse(value)
                        for key, value in json_separation["S_label"].items()
                    },
                    "S_star": {
                        key: value
                        for key, value in json_separation["S_star"].items()
                    },
                    "sigma": {
                        key: value
                        for key, value in json_separation["sigma"].items()
                    },
                    "separated": {
                        key: [tuple(pair) for pair in value]
                        for key, value in json_separation["separated"].items()
                    },
                    "kept": {
                        key: [tuple(pair) for pair in value]
                        for key, value in json_separation["kept"].items()
                    }
                }

        main(args.filename, pairs, separation)
