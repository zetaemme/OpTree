import logging
from argparse import ArgumentParser, BooleanOptionalAction
from os.path import dirname
from pathlib import Path
from pickle import HIGHEST_PROTOCOL, Unpickler, dump
from timeit import timeit

from src.dataset import Dataset
from src.decision_tree import build_decision_tree
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
    build_decision_tree(dataset)


def main(dataset_path: str, dataset_pairs: PicklePairs | None, dataset_separation: PickleSeparation | None) -> None:
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
    with open(f"model/decision_tree_{dataset_name}.pkl", "wb") as obj_file:
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
            print("HEREEE")
            with open(dirname(__file__) + f"/data/separation/{dataset_name}_separation.pkl", "rb") as separation_f:
                logger.info("Loading separation from Pickle file")
                unpickler = Unpickler(separation_f)
                separation: PickleSeparation = unpickler.load()

        main(args.filename, pairs, separation)
