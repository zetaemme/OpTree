import logging

from src.budget import find_budget
from src.dataset import Dataset
from src.extraction import cheapest_separation, eligible_labels
from src.maximization import pairs_maximization, probability_maximization
from src.tree import Tree

logger = logging.getLogger(__name__)


def build_decision_tree(dataset: Dataset, decision_tree=Tree(), last_added_node: str = None) -> tuple[Tree, bool]:
    """Recursively builds a (log)-optimal decision tree.

    Args:
        dataset (Dataset): The dataset used to train the model
        decision_tree (Tree): The tree to build
        last_added_node (str): Last node added to the tree. Defaults to None

    Returns:
        Tree: The (log)-optimal decision tree
    """
    # BASE CASE: If no pairs, return a leaf labelled by a class
    if dataset.pairs_number == 0:
        tree = Tree()

        # NOTE: Avoids insertion of a wrong leaf when the dataset contains a value with just the "index" column
        if dataset.features:
            leaf = list(dataset.classes.values())[0]
            logger.info("No pairs in dataset, setting leaf \"%s\"", leaf)
            tree.add_leaf(leaf, leaf)
        else:
            logger.info("No more objects in dataset")

        return tree, False

    # BASE CASE: If just one pair
    if dataset.pairs_number == 1:
        logger.info(f"Just one pair in dataset: {dataset.pairs_list[0]}")
        # Create a tree rooted by the cheapest test that separates the two items
        terminal_tree = Tree()
        split = cheapest_separation(dataset, dataset.pairs_list[0][0], dataset.pairs_list[0][1])

        logger.info("Setting node \"%s\" as root of subtree", split)
        terminal_tree.add_node(None, split, split)

        # Add the two items as leafs labelled with the respective class
        class_1 = dataset.classes[dataset.pairs_list[0][0]]
        label_1 = str(dataset[0, dataset.features.index(split) + 1])
        class_2 = dataset.classes[dataset.pairs_list[0][1]]
        label_2 = str(dataset[1, dataset.features.index(split) + 1])

        logger.info("Adding leaf \"%s\"", class_1)
        terminal_tree.add_leaf(class_1, label_1)
        logger.info("Adding leaf \"%s\"", class_2)
        terminal_tree.add_leaf(class_2, label_2)

        dataset.drop_feature(split)

        return terminal_tree, True

    budget = find_budget(dataset)
    logger.info("Using budget %f", budget)

    spent = 0.0
    spent_2 = 0.0

    universe = dataset.copy()

    k = 1
    chosen_test = ""

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
            decision_tree.add_node(None, chosen_test, chosen_test)
            last_added_node = decision_tree.root["id"]
        else:
            # Set chosen_test as child of the test added in the last iteration
            decision_tree.add_node(last_added_node, chosen_test, "t_A ?")

        # For each label in the possible outcomes of chosen_test
        for label in eligible_labels(universe, chosen_test):
            logger.info("Label %s for test \"%s\"", label, chosen_test)
            universe_intersection = universe.intersection(universe.S_label[chosen_test][label])
            logger.debug(f"Universe intersect S[{chosen_test}][{label}]: {universe_intersection.indexes}")

            # Set the tree resulting from the recursive call as the child of chosen_test
            logger.info("t_A recursive call with test \"%s\"", chosen_test)
            subtree, is_split_base_case = build_decision_tree(universe_intersection, decision_tree, chosen_test)

            if is_split_base_case and subtree.root["id"] in universe.features:
                # universe.drop_feature(subtree.root["id"])
                del budgeted_features[subtree.root["id"]]

            decision_tree.add_subtree(chosen_test, subtree, label)

        universe = universe.intersection(universe.S_star[chosen_test])
        spent += universe.costs[chosen_test]
        universe.drop_feature(chosen_test)
        del budgeted_features[chosen_test]
        k += 1

    logger.info("End of t_A part of the procedure!")

    # If there are still some tests with cost greater than budget
    if budgeted_features:
        while True:
            chosen_test = pairs_maximization(universe)
            logger.debug("Chosen test: %s", chosen_test)

            # Set chosen_test as child of the test added in the last iteration
            # FIXME: Che label metto?
            decision_tree.add_node(last_added_node, chosen_test, chosen_test)

            # For each label in the possible outcomes of chosen_test
            for label in eligible_labels(universe, chosen_test):
                logger.info("Label %s for test \"%s\"", label, chosen_test)
                universe_intersection = universe.intersection(universe.S_label[chosen_test][label])
                logger.debug(f"Universe intersect S[{chosen_test}][{label}]: {universe_intersection.indexes}")

                # Set the tree resulting from the recursive call as the child of chosen_test
                logger.info("t_B recursive call with test \"%s\"", chosen_test)
                subtree, is_split_base_case = build_decision_tree(universe_intersection, decision_tree)

                if is_split_base_case and subtree.root["id"] in universe.features:
                    # universe.drop_feature(subtree.root["id"])
                    del budgeted_features[subtree.root["id"]]

                decision_tree.add_subtree(chosen_test, subtree, label)

            universe = universe.intersection(universe.S_star[chosen_test])
            spent_2 += dataset.costs[chosen_test]
            universe.drop_feature(chosen_test)
            del budgeted_features[chosen_test]
            k += 1

            # If there are no tests left, or we're running out of budget, break the loop
            if budget - spent_2 < 0 or not budgeted_features:
                break

    logger.info("End of t_B part of the procedure!")

    # Set the tree resulting from the recursive call as child of the test added in the last iteration
    logger.info("Final recursive call")
    subtree, _ = build_decision_tree(universe, decision_tree, chosen_test)
    decision_tree.add_subtree(chosen_test, subtree, "t_F ?")

    return decision_tree, False
