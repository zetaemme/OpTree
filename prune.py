from argparse import ArgumentParser
from os.path import dirname
from pathlib import Path
from pickle import HIGHEST_PROTOCOL, Unpickler, dump

from src.dataset import Dataset
from src.decision_tree import DecisionTree
from src.utils import prune

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="prune.py",
        description="Prunes the tree",
    )
    parser.add_argument(
        "-t",
        "--tree",
        type=str,
        help="The Pickle file containing the model",
    )
    parser.add_argument(
        "-td",
        "--train_dataset",
        type=str,
        help="The Pickle file containing the train dataset",
    )

    args = parser.parse_args()

    tree_path = args.tree
    if Path(dirname(__file__) + f"/{tree_path}").is_file():
        with open(dirname(__file__) + f"/{tree_path}", "rb") as tree_file:
            unpickler = Unpickler(tree_file)
            tree = unpickler.load()

    dataset_path = args.train_dataset
    if Path(dirname(__file__) + f"/{dataset_path}").is_file():
        with open(dirname(__file__) + f"/{dataset_path}", "rb") as data_file:
            unpickler = Unpickler(data_file)
            dataset: Dataset = unpickler.load()

    dataset_name, fold_number = dataset_path.split("/")[1:3]

    pruned = prune(tree, dataset)
    pruned_tree = DecisionTree()
    pruned_tree.decision_tree = pruned
    pruned_tree.dataset = dataset

    with open(dirname(__file__) + f"/model/{dataset_name}/{fold_number}/pruned.pkl", "wb") as pruned_file:
        dump(pruned_tree, pruned_file, HIGHEST_PROTOCOL)
