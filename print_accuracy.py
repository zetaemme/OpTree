from argparse import ArgumentParser
from collections import Counter
from os.path import dirname
from pathlib import Path
from pickle import Unpickler

from src.dataset import Dataset
from src.decision_tree import DecisionTree

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="rocauc.py",
        description="Plots the ROC AUC curve for a given decision tree model",
    )
    parser.add_argument(
        "-t",
        "--tree",
        type=str,
        help="The Pickle file containing the model's best fit",
    )
    parser.add_argument(
        "-td",
        "--test_dataset",
        type=str,
        help="The Pickle file containing the test dataset",
    )

    args = parser.parse_args()

    tree_path = args.tree
    if Path(dirname(__file__) + f"/{tree_path}").is_file():
        with open(dirname(__file__) + f"/{tree_path}", "rb") as bf_file:
            unpickler = Unpickler(bf_file)
            decision_tree: DecisionTree = unpickler.load()

    td_path = args.test_dataset
    if Path(dirname(__file__) + f"/{td_path}").is_file():
        with open(dirname(__file__) + f"/{td_path}", "rb") as td_file:
            unpickler = Unpickler(td_file)
            test_dataset: Dataset = unpickler.load()

    results = []
    predictions = []
    for row in test_dataset.data(True):
        correct = str(row[-1])
        prediction = decision_tree.predict(row[1:-1])

        predictions.append(prediction)
        results.append(prediction == correct)

    counter = Counter(results)
    print(f"Accuracy: {(counter[True] / len(results)) * 100:.2f}%")
    print(f"Number of nodes: {decision_tree.number_of_nodes()}")
    print(f"Height: {decision_tree.height()}")
