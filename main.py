# from treelib import Tree
import argparse
import logging
from os.path import dirname
from pathlib import Path

from src.dataset import Dataset
from src.decision_tree import build_decision_tree
from src.separation import Separation

DEBUG = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - OpSion @ {%(module)s#%(funcName)s} - %(message)s",
    datefmt="%H:%M:%S"
)


def main(dataset_path: str) -> None:
    """Inits dataset and test list in order to pass them to the algorithm"""
    path = Path(dirname(__file__) + "/" + dataset_path)
    dataset = Dataset(path)
    separation = Separation(dataset)

    build_decision_tree(dataset, separation)

    # joblib.dump(decision_tree, 'model/dectree.sav')
    # decision_tree.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Builds (log-)optimal decision trees"
    )
    parser.add_argument(
        "filename",
        type=str,
        help="The CSV file containing the dataset"
    )

    args = parser.parse_args()

    main(args.filename)
