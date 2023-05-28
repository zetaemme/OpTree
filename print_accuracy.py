from argparse import ArgumentParser
from collections import Counter
from os.path import dirname
from pathlib import Path
from pickle import Unpickler

from src.dataset import Dataset
from src.decision_tree import DecisionTree

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="print_accuracy.py",
        description="Print statistics for the given dataset",
    )
    parser.add_argument(
        "-dn",
        "--dataset_name",
        type=str,
        help="The dataset name",
    )

    args = parser.parse_args()

    accuracies: dict = {}
    for i in range(1, 6):
        if Path(dirname(__file__) + f"/model/{args.dataset_name}/{i}/pruned.pkl").is_file():
            with open(dirname(__file__) + f"/model/{args.dataset_name}/{i}/pruned.pkl", "rb") as tree_file:
                unpickler = Unpickler(tree_file)
                decision_tree: DecisionTree = unpickler.load()

        if Path(dirname(__file__) + f"/model/{args.dataset_name}/{i}/test_set.pkl").is_file():
            with open(dirname(__file__) + f"/model/{args.dataset_name}/{i}/test_set.pkl", "rb") as td_file:
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

        accuracy = (counter[True] / len(results)) * 100
        accuracies[i] = round(accuracy, 2)
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Number of nodes: {decision_tree.number_of_nodes()}")
        print(f"Height: {decision_tree.height()}")

    print(accuracies)
