import logging
from argparse import ArgumentParser
from collections import Counter
from math import ceil
from os.path import dirname
from pathlib import Path
from pickle import HIGHEST_PROTOCOL, Unpickler, dump
from statistics import mean
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

    # NOTE: 31/03/2023 - Since it happens to have structurally equal objects with different class label, we remove
    #                    the most represented one. Doing so, we assure the purity of the leaves while keeping intact
    #                    the variance of data
    dataset.drop_equal_objects_with_different_class()

    folds_accuracies = {}
    folds_nodes = {}
    folds_heights = {}

    # Computes the indexes for each of the K folds
    k_folded_dataset = dataset.k_fold_split(5)

    for fold_number, fold in enumerate(k_folded_dataset, 1):
        logger.info("Training on fold: %i", fold_number)
        src.TESTS = fold["train"].features
        src.COSTS = fold["train"].costs

        with open(dirname(__file__) + f"/model/{name}/{fold_number}/train_set.pkl", "wb") as test_file:
            dump(fold["train"], test_file, HIGHEST_PROTOCOL)

        # Fits the decision tree on the i-th training fold
        decision_tree = DecisionTree()
        decision_tree.fit(fold["train"], src.TESTS, src.COSTS, name, fold_number)

        with open(dirname(__file__) + f"/model/{name}/{fold_number}/pruned.pkl", "wb") as tree_file:
            dump(decision_tree, tree_file, HIGHEST_PROTOCOL)

        with open(dirname(__file__) + f"/model/{name}/{fold_number}/test_set.pkl", "wb") as test_file:
            dump(fold["test"], test_file, HIGHEST_PROTOCOL)

        results = []
        predictions = []
        # Tests the previously fitted tree on the i-th test fold
        for row in fold["test"].data(True):
            correct = str(row[-1])
            prediction = decision_tree.predict(row[1:-1])

            predictions.append(prediction)
            results.append(prediction == correct)

        # Computes the metrics for the i-th fold
        counter = Counter(results)
        folds_accuracies[fold_number] = (counter[True] / len(results)) * 100
        folds_nodes[fold_number] = decision_tree.number_of_nodes()
        folds_heights[fold_number] = decision_tree.height()

    logger.info("Mean accuracy over 5 folds: %.2f", mean(folds_accuracies.values()))
    logger.info("Mean number of nodes over 5 folds: %i", ceil(mean(folds_nodes.values())))
    logger.info("Mean height over 5 folds: %i", ceil(mean(folds_heights.values())))
    logger.info("Best fold: %i", max(folds_accuracies, key=folds_accuracies.get))


if __name__ == "__main__":
    parser = ArgumentParser(prog="main.py", description="Builds (log-)optimal decision trees")
    parser.add_argument("-f", "--filename", type=str, help="The CSV file containing the dataset")

    args = parser.parse_args()

    dataset_name = args.filename.replace('data/datasets/csv/', '').replace('.csv', '')
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
