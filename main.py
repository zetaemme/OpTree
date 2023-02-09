import logging
from argparse import ArgumentParser
from os.path import dirname
from pathlib import Path
from pickle import HIGHEST_PROTOCOL, dump

from src import DEBUG
from src.dataset import Dataset
from src.decision_tree import build_decision_tree
from src.separation import Separation

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] - OpSion @ {%(module)s.py -> %(funcName)s} - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(dataset_path: str) -> None:
    """Inits dataset and test list in order to pass them to the algorithm"""
    path = Path(dirname(__file__) + "/" + dataset_path)
    dataset = Dataset(path)
    separation = Separation(dataset)

    decision_tree = build_decision_tree(dataset, separation)

    logger.info("Done!")
    decision_tree.print()

    if decision_tree is not None:
        with open("model/decision_tree.obj", "wb") as obj_file:
            dump(decision_tree, obj_file, HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = ArgumentParser(prog="main.py", description="Builds (log-)optimal decision trees")
    parser.add_argument("-f", "--filename", type=str, help="The CSV file containing the dataset")

    args = parser.parse_args()
    main(args.filename)
