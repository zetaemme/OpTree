from treelib import Node, Tree

from src.budget import find_budget
from src.dataset import Dataset
from src.extraction import cheapest_separation
from src.separation import Separation


def build_decision_tree(dataset: Dataset, separation: Separation) -> Tree:
    if dataset.pairs_number == 0:
        decision_tree: Tree = Tree()
        decision_tree.create_node(dataset.classes[0])

        return decision_tree

    if dataset.pairs_number == 1:
        decision_tree: Tree = Tree()

        root: Node = Node(cheapest_separation(
            dataset,
            dataset.pairs_list[0][0],
            dataset.pairs_list[0][1],
            separation
        ))

        decision_tree.add_node(root)
        decision_tree.create_node(dataset.get_class(dataset.pairs_list[0][0]), parent=root)
        decision_tree.create_node(dataset.get_class(dataset.pairs_list[0][1]), parent=root)

    budget: float = find_budget(dataset)

    spent: float = 0
    spent_2: float = 0

    # FIXME: Maybe is an expensive copy, try to use numpy copy() method
    universe: Dataset = dataset

    k: int = 1

    budgeted_features: list = [test for test, cost in dataset.costs.items() if cost <= budget]

    while any(test <= budget - spent for test in budgeted_features):
        pass
