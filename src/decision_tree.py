import logging
from src.budget import find_budget
from src.dataset import Dataset
from src.extraction import cheapest_separation
from src.maximization import pairs_maximization, probability_maximization
from src.separation import Separation
from src.tree import Tree

logger = logging.getLogger(__name__)


def build_decision_tree(
    dataset: Dataset, separation: Separation, decision_tree=Tree(None)
) -> Tree:
    """Recursively builds a (log)-optimal decision tree.

    Args:
        dataset (Dataset): The dataset used to train the model
        separation (Separation): Dataset tripartition and sets
        decision_tree (Tree): The tree to build

    Returns:
        Tree: The (log)-optimal decision tree
    """
    # BASE CASE: If no pairs, return a leaf labelled by a class
    if dataset.pairs_number == 0:
        return Tree(dataset.classes[0])

    # BASE CASE: If just one pair
    if dataset.pairs_number == 1:
        # Create a tree labelled by the cheapest test that separates the two items
        terminal_tree = Tree(
            cheapest_separation(
                dataset, dataset.pairs_list[0][0], dataset.pairs_list[0][1], separation
            )
        )

        # Add the two items as leafs labelled with the respective class
        terminal_tree.add_children(
            [
                Tree(dataset.classes[dataset.pairs_list[0][0]]),
                Tree(dataset.classes[dataset.pairs_list[0][1]]),
            ]
        )

        return terminal_tree

    budget = find_budget(dataset, separation)
    logger.info("Using budget %f", budget)

    spent = 0.0
    spent_2 = 0.0

    universe = dataset.copy()

    k = 1

    # Removes from T all tests with cost greater than budget
    budgeted_features = {
        test: cost for test, cost in dataset.costs.items() if cost <= budget
    }
    logger.info(f"Features within budget {list(budgeted_features.keys())}")

    # While exists at least a test with cost equal or less than (budget - spent)
    while any(cost <= budget - spent for cost in budgeted_features.values()):
        chosen_test = probability_maximization(universe, budget, spent)
        logger.debug("Chosen test: %s", chosen_test)

        if k == 1:
            # Set chosen_test as the root of the tree
            decision_tree.set_label(chosen_test)
        else:
            # Set chosen_test as child of the test added in the last iteration
            decision_tree.add_child(Tree(chosen_test))

        # For each object in the possible outcomes of chosen_test
        for objects in separation.S_label[chosen_test].values():
            universe_intersection = universe.intersection(objects)
            logger.debug(f"Intersection with {objects}: {universe_intersection}")

            if universe_intersection and objects != separation.S_star[chosen_test]:
                # Set the tree resulting from the recursive call as the child of chosen_test
                decision_tree.last_added.add_child(
                    build_decision_tree(
                        universe_intersection,
                        # FIXME: Separation andrebbe ricalcolato (?)
                        separation,
                        decision_tree,
                    )
                )

        universe = universe.intersection(separation.S_star[chosen_test])
        spent += dataset.costs[chosen_test]
        del budgeted_features[chosen_test]
        k += 1

    # If there are still some tests with cost greater than budget
    if budgeted_features:
        while True:
            chosen_test = pairs_maximization(universe)
            logger.debug("Chosen test: %s", chosen_test)

            # Set chosen_test as child of the test added in the last iteration
            decision_tree.add_child(Tree(chosen_test))

            # For each object in the possible outcomes of chosen_test
            for objects in separation.S_label[chosen_test].values():
                universe_intersection = universe.intersection(objects)
                logger.debug(f"Intersection with {objects}: {universe_intersection}")

                if universe_intersection and objects != separation.S_star[chosen_test]:
                    # Set the tree resulting from the recursive call as the child of chosen_test
                    decision_tree.last_added.add_child(
                        build_decision_tree(
                            universe_intersection,
                            # FIXME: Separation andrebbe ricalcolato (?)
                            separation,
                            decision_tree,
                        )
                    )

            universe = universe.intersection(separation.S_star[chosen_test])
            spent_2 += dataset.costs[chosen_test]
            del budgeted_features[chosen_test]
            k += 1

            # If there are no tests left or we're running out of budget, break the loop
            if budget - spent_2 < 0 or not budgeted_features:
                break

    # Set the tree resulting from the recursive call as child of the test added in the last iteration
    decision_tree.add_child(
        # FIXME: Separation andrebbe ricalcolato (?)
        build_decision_tree(universe, separation, decision_tree)
    )

    return decision_tree
