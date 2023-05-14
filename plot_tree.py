from argparse import ArgumentParser
from os.path import dirname
from pickle import Unpickler

from src.decision_tree import DecisionTree

if __name__ == "__main__":
    parser = ArgumentParser(prog="plot_tree.py", description="Plots a given decision tree model")
    parser.add_argument("-t", "--tree", type=str, help="The Pickle file containing the model")

    args = parser.parse_args()

    tree_path = args.tree

    with open(dirname(__file__) + f"/{tree_path}", "rb") as tree_file:
        unpickler = Unpickler(tree_file)
        tree: DecisionTree = unpickler.load()

    tree.print()
