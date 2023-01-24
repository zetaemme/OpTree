from src.budget import find_budget
from src.dataset import Dataset
from src.extraction import cheapest_separation
from src.maximization import pairs_maximization, probability_maximization
from src.separation import Separation
from src.tree import Tree
from src.utils import get_parent_node


def build_decision_tree(
    dataset: Dataset, separation: Separation, decision_tree=Tree(None)
) -> Tree:
    """Recursively builds a (log)-optimal decision tree.

    Args:
        dataset (Dataset): The dataset used to train the model
        separation (Separation): Dataset tripartition and sets

    Returns:
        Tree: The (log)-optimal decision tree
    """
    if dataset.pairs_number == 0:
        return Tree(dataset.classes[0])

    if dataset.pairs_number == 1:
        terminal_tree = Tree(
            cheapest_separation(
                dataset, dataset.pairs_list[0][0], dataset.pairs_list[0][1], separation
            )
        )

        terminal_tree.add_child(
            Tree(dataset.get_class_of(dataset.pairs_list[0][0])),
        )
        terminal_tree.add_child(
            Tree(dataset.get_class_of(dataset.pairs_list[0][1])),
        )

        return terminal_tree

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

        if k == 1:
            decision_tree.set_label(chosen_test)
        else:

            decision_tree.add_child(
                Tree(chosen_test),
                # FIXME: Utilizzare get parent node per inserire in modo corretto il nuovo nodo
                parent_label=get_parent_node(list(budgeted_features), chosen_test),
            )

        for objects in separation.S_label[chosen_test].values():
            universe_intersection = universe.intersection(objects)

            if universe_intersection and objects != separation.S_star[chosen_test]:
                decision_tree.add_child(
                    build_decision_tree(
                        universe_intersection,
                        # FIXME: Separation andrebbe ricalcolato (?)
                        separation,
                        decision_tree,
                    ),
                    # FIXME: Utilizzare get parent node per inserire in modo corretto il nuovo nodo
                    parent_label=get_parent_node(list(budgeted_features), chosen_test),
                )

        universe = universe.intersection(separation.S_star[chosen_test])
        spent += dataset.costs[chosen_test]
        del budgeted_features[chosen_test]
        k += 1

    if budgeted_features:
        while True:
            chosen_test = pairs_maximization(universe)

            decision_tree.add_child(
                Tree(chosen_test),
                # FIXME: Utilizzare get parent node per inserire in modo corretto il nuovo nodo
                parent_label=get_parent_node(list(budgeted_features), chosen_test),
            )

            for objects in separation.S_label[chosen_test].values():
                universe_intersection = universe.intersection(objects)

                if universe_intersection and objects != separation.S_star[chosen_test]:
                    decision_tree.add_child(
                        build_decision_tree(
                            universe_intersection,
                            # FIXME: Separation andrebbe ricalcolato (?)
                            separation,
                            decision_tree,
                        ),
                        # FIXME: Utilizzare get parent node per inserire in modo corretto il nuovo nodo
                        parent_label=get_parent_node(
                            list(budgeted_features), chosen_test
                        ),
                    )

            universe = universe.intersection(separation.S_star[chosen_test])
            spent_2 += dataset.costs[chosen_test]
            del budgeted_features[chosen_test]
            k += 1

            if budget - spent_2 < 0 or not budgeted_features:
                break

    decision_tree.add_child(
        # FIXME: Separation andrebbe ricalcolato (?)
        build_decision_tree(universe, separation, decision_tree),
        parent_label=# Nodo corrispondente a k - 1
    )

    return decision_tree
