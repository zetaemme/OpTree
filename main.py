import logging
from argparse import ArgumentParser, BooleanOptionalAction
from os.path import dirname
from pathlib import Path
from pickle import HIGHEST_PROTOCOL, dump
from timeit import timeit

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


def main(dataset_path: str) -> None:
    """Inits dataset and runs the algorithm"""
    path = Path(dirname(__file__) + "/" + dataset_path)
    dataset = Dataset(path)

    decision_tree = build_decision_tree(dataset)

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
        main(args.filename)
