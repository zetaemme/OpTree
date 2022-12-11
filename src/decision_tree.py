from treelib import Node, Tree

from src.budget import find_budget
from src.dataset import Dataset
from src.extraction import cheapest_separation
from src.maximization import pairs_maximization, probability_maximization
from src.separation import Separation
from src.utils import get_parent_node


def build_decision_tree(
    dataset: Dataset, separation: Separation, decision_tree=Tree()
) -> Tree:
    """Recursively builds an optimal decision tree.

    Args:
        dataset (Dataset): The dataset used to train the model
        separation (Separation): Dataset tripartition and sets

    Returns:
        Tree: The optimal decision tree
    """
    if dataset.pairs_number == 0:
        decision_tree.create_node(dataset.classes[0])

        return decision_tree

    if dataset.pairs_number == 1:
        root = Node(
            cheapest_separation(
                dataset, dataset.pairs_list[0][0], dataset.pairs_list[0][1], separation
            )
        )

        decision_tree.add_node(root)
        decision_tree.create_node(
            dataset.get_class_of(dataset.pairs_list[0][0]), parent=root
        )
        decision_tree.create_node(
            dataset.get_class_of(dataset.pairs_list[0][1]), parent=root
        )

    # FIXME: Da fixare euristica
    budget = find_budget(dataset, separation)

    spent = 0.0
    spent_2 = 0.0

    universe = dataset.copy()

    k = 1

    budgeted_features = {
        test: cost for test, cost in dataset.costs.items() if cost <= budget
    }

    while any(cost <= budget - spent for cost in budgeted_features.values()):
        chosen_test = probability_maximization(universe, budget, spent)

        # NOTE: Is it possible to sort the tests before the loop, resulting in a simple "for each" iteration (?)
        if list(budgeted_features).index(chosen_test) == 0:
            decision_tree.create_node(chosen_test, identifier=chosen_test)
        else:
            decision_tree.create_node(
                chosen_test,
                identifier=chosen_test,
                parent=get_parent_node(list(budgeted_features), chosen_test),
            )

        # TODO: For every...

        universe = universe.intersection(separation.S_star[chosen_test])
        spent += dataset.costs[chosen_test]
        del budgeted_features[chosen_test]
        k += 1

    if budgeted_features:
        while True:
            chosen_test = pairs_maximization(universe)

            decision_tree.create_node(
                chosen_test,
                identifier=chosen_test,
                parent=get_parent_node(list(budgeted_features), chosen_test),
            )

            # TODO: For every...

            universe = universe.intersection(separation.S_star[chosen_test])
            spent_2 += dataset.costs[chosen_test]
            del budgeted_features[chosen_test]
            k += 1

            if budget - spent_2 < 0 or not budgeted_features:
                break

    # FIXME: Separation andrebbe ricalcolato (?)
    sub_decision_tree = build_decision_tree(universe, separation, decision_tree)
    # TODO: Make sub_decision_tree child of decision_tree

    return decision_tree
