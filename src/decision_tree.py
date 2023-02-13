import logging

from src.budget import find_budget
from src.dataset import Dataset
from src.extraction import cheapest_separation, eligible_labels
from src.maximization import pairs_maximization, probability_maximization
from src.tree import Tree

logger = logging.getLogger(__name__)


def build_decision_tree(dataset: Dataset, decision_tree=Tree()) -> Tree:
    """Recursively builds a (log)-optimal decision tree.

    Args:
        dataset (Dataset): The dataset used to train the model
        decision_tree (Tree): The tree to build

    Returns:
        Tree: The (log)-optimal decision tree
    """
    # BASE CASE: If no pairs, return a leaf labelled by a class
    if dataset.pairs_number == 0:
        tree = Tree()

        # NOTE: Avoids insertion of a wrong leaf when the dataset contains a value with just the "index" column
        if dataset.features:
            tree.add_leaf(dataset.classes[0], dataset.classes[0])

        return tree

    # BASE CASE: If just one pair
    if dataset.pairs_number == 1:
        # Create a tree rooted by the cheapest test that separates the two items
        terminal_tree = Tree()
        split = cheapest_separation(dataset, dataset.pairs_list[0][0], dataset.pairs_list[0][1])

        terminal_tree.add_node(split, split)

        # Add the two items as leafs labelled with the respective class
        class_1 = dataset.classes[dataset.pairs_list[0][0]]
        label_1 = str(dataset[0, dataset.features.index(split) + 1])
        class_2 = dataset.classes[dataset.pairs_list[0][1]]
        label_2 = str(dataset[1, dataset.features.index(split) + 1])

        terminal_tree.add_leaf(class_1, label_1)
        terminal_tree.add_leaf(class_2, label_2)

        dataset.drop_feature(split)

        return terminal_tree

    budget = find_budget(dataset)
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
            decision_tree.add_node(chosen_test, chosen_test)
        else:
            # Set chosen_test as child of the test added in the last iteration
            # FIXME: Serve la label che inserisce "2" come edge per "t3"
            decision_tree.add_node(chosen_test, "2")

        # For each label in the possible outcomes of chosen_test
        for label in eligible_labels(universe, chosen_test):
            universe_intersection = universe.intersection(universe.S_label[chosen_test][label])
            logger.debug(f"Universe intersect S[{chosen_test}][{label}]: {universe_intersection.indexes}")

            # Set the tree resulting from the recursive call as the child of chosen_test
            logger.info("t_A recursive call")
            decision_tree.add_subtree(
                build_decision_tree(
                    universe_intersection,
                    decision_tree,
                ),
                label,
                True
            )

        universe = universe.intersection(universe.S_star[chosen_test])
        spent += universe.costs[chosen_test]
        universe.drop_feature(chosen_test)
        del budgeted_features[chosen_test]
        k += 1

    # If there are still some tests with cost greater than budget
    if budgeted_features:
        while True:
            chosen_test = pairs_maximization(universe)
            logger.debug("Chosen test: %s", chosen_test)

            # Set chosen_test as child of the test added in the last iteration
            # FIXME: Che label metto?
            decision_tree.add_node(chosen_test, chosen_test)

            # For each label in the possible outcomes of chosen_test
            for label in eligible_labels(universe, chosen_test):
                universe_intersection = universe.intersection(universe.S_label[chosen_test][label])
                logger.debug(f"Universe intersect S[{chosen_test}][{label}]: {universe_intersection.indexes}")

                # Set the tree resulting from the recursive call as the child of chosen_test
                logger.info("t_B recursive call")
                decision_tree.add_subtree(
                    build_decision_tree(
                        universe_intersection,
                        decision_tree
                    ),
                    label,
                    False
                )

            universe = universe.intersection(universe.S_star[chosen_test])
            spent_2 += dataset.costs[chosen_test]
            universe.drop_feature(chosen_test)
            del budgeted_features[chosen_test]
            k += 1

            # If there are no tests left, or we're running out of budget, break the loop
            if budget - spent_2 < 0 or not budgeted_features:
                break

    # Set the tree resulting from the recursive call as child of the test added in the last iteration
    logger.info("Final recursive call")
    decision_tree.add_subtree(
        build_decision_tree(universe, decision_tree),
        # FIXME: Che label metto?
        "FINAL",
        False
    )

    return decision_tree
