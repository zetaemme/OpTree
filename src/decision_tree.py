from treelib import Node, Tree

from src.budget import find_budget
from src.dataset import Dataset
from src.extraction import cheapest_separation
from src.separation import Separation


def build_decision_tree(dataset: Dataset, separation: Separation) -> Tree:
    if dataset.pairs_number == 0:
        decision_tree = Tree()
        decision_tree.create_node(dataset.classes[0])

        return decision_tree

    if dataset.pairs_number == 1:
        decision_tree = Tree()

        root = Node(cheapest_separation(
            dataset,
            dataset.pairs_list[0][0],
            dataset.pairs_list[0][1],
            separation
        ))

        decision_tree.add_node(root)
        decision_tree.create_node(dataset.get_class_of(dataset.pairs_list[0][0]), parent=root)
        decision_tree.create_node(dataset.get_class_of(dataset.pairs_list[0][1]), parent=root)

    budget: float = find_budget(dataset, separation)

    spent = 0.0
    spent_2 = 0.0

    universe = dataset.copy()

    k: int = 1

    budgeted_features = {test: cost for test, cost in dataset.costs.items() if cost <= budget}

    while any(cost <= budget - spent for cost in budgeted_features.values()):
        pass

        if k == 1:
            pass
