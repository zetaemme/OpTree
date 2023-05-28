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
        "-dn",
        "--dataset_name",
        type=str,
        help="The name of the dataset",
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        help="The cutoff parameter",
    )

    args = parser.parse_args()

    for i in range(1, 6):

        if Path(dirname(__file__) + f"/model/{args.dataset_name}/{i}/not_pruned.pkl").is_file():
            with open(dirname(__file__) + f"/model/{args.dataset_name}/{i}/not_pruned.pkl", "rb") as tree_file:
                unpickler = Unpickler(tree_file)
                tree = unpickler.load()

        if Path(dirname(__file__) + f"/model/{args.dataset_name}/{i}/train_set.pkl").is_file():
            with open(dirname(__file__) + f"/model/{args.dataset_name}/{i}/train_set.pkl", "rb") as data_file:
                unpickler = Unpickler(data_file)
                dataset: Dataset = unpickler.load()

        pruned = prune(tree, dataset, args.epsilon)
        pruned_tree = DecisionTree()
        pruned_tree.decision_tree = pruned
        pruned_tree.dataset = dataset

        with open(dirname(__file__) + f"/model/{args.dataset_name}/{i}/pruned.pkl", "wb") as pruned_file:
            dump(pruned_tree, pruned_file, HIGHEST_PROTOCOL)
