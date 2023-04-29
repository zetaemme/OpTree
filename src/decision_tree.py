import logging
from typing import Optional
from uuid import UUID

import src
from src.budget import find_budget
from src.dataset import Dataset
from src.extraction import cheapest_separation, eligible_labels
from src.maximization import pairs_maximization, probability_maximization
from src.tree import Tree
from src.utils import get_backbone_label

logger = logging.getLogger(__name__)


def build_decision_tree(
        dataset: Dataset,
        tests: list[str],
        costs: dict[str, float],
        decision_tree: Tree,
        last_added_node: Optional[UUID] = None
) -> tuple[Tree, bool]:
    """Recursively builds a (log)-optimal decision tree.

    Args:
        dataset (Dataset): The dataset used to train the model
        tests (list[str]): The features from which the tree will be built
        costs (doct[str, float]): The costs for the tests
        decision_tree (Tree): The tree to build
        last_added_node (str): Last node added to the tree. Defaults to None

    Returns:
        Tree: The (log)-optimal decision tree
    """
    # BASE CASE: If no pairs, return a leaf labelled by a class
    if dataset.pairs_number == 0:
        tree = Tree()

        # NOTE: Avoids insertion needless leaf
        if dataset.features and len(dataset) != 0:
            leaf = list(dataset.classes.values())[0]
            logger.info("No pairs in dataset, setting leaf \"%s\"", leaf)
            tree.add_node(leaf)
        else:
            logger.info("No more objects in dataset")

        return tree, False

    # BASE CASE: If just one pair
    if dataset.pairs_number == 1:
        logger.info(f"Just one pair in dataset: {dataset.pairs_list[0]}")

        # Create a tree rooted by the cheapest test that separates the two items
        tree = Tree()
        split = cheapest_separation(dataset, costs, dataset.pairs_list[0])

        logger.info("Setting node \"%s\" as root of subtree", split)
        root_id = tree.add_node(split)

        # Add the two items as leafs labelled with the respective class
        class_1 = dataset.classes[dataset.pairs_list[0][0]]
        label_1 = str(dataset[0, dataset.features.index(split) + 1])
        class_2 = dataset.classes[dataset.pairs_list[0][1]]
        label_2 = str(dataset[1, dataset.features.index(split) + 1])

        logger.info("Adding leaf \"%s\"", class_1)
        tree.add_node(class_1, root_id, label_1)
        logger.info("Adding leaf \"%s\"", class_2)
        tree.add_node(class_2, root_id, label_2)

        return tree, True

    budget = find_budget(dataset, tests, src.COSTS)
    logger.info("Using budget %f", budget)

    spent = 0.0
    spent_2 = 0.0

    universe = dataset.copy()

    # Removes from T all tests with cost greater than budget
    budgeted_features = [test for test in tests if costs[test] <= budget]
    logger.info(f"{len(budgeted_features)} features within budget: {budgeted_features}")

    # While exists at least a test with cost equal or less than (budget - spent)
    logger.info("Starting t_A part of the procedure")
    while any(cost <= budget - spent for test, cost in costs.items() if test in budgeted_features) and len(
            universe) != 0:
        chosen_test = probability_maximization(universe, budgeted_features, costs, budget, spent)
        logger.debug("Chosen test: %s", chosen_test)

        if decision_tree.is_empty:
            # Set chosen_test as the root of the tree
            logger.info("Setting %s as root", chosen_test)
            last_added_node = decision_tree.add_node(chosen_test)
        else:
            # Set chosen_test as child of the test added in the last iteration
            backbone_label = get_backbone_label(universe, chosen_test)

            logger.info("Adding node %s", chosen_test)
            last_added_node = decision_tree.add_node(chosen_test, last_added_node, backbone_label)

        # For each label in the possible outcomes of chosen_test
        for label in eligible_labels(universe, chosen_test):
            logger.info("Label %s for test \"%s\"", label, chosen_test)
            universe_intersection = universe.intersection(universe.S_label[chosen_test][label])
            logger.debug(f"Universe intersect S[{chosen_test}][{label}]: {universe_intersection.indexes}")

            # Set the tree resulting from the recursive call as the child of chosen_test
            logger.info("t_A recursive call with test \"%s\"", chosen_test)
            subtree, is_split_base_case = build_decision_tree(
                # NOTE: 27/02/2023 - Remove the chosen feature before the recursive call
                #       Instead of removing it from the dataset just to add it back after the return an updated copy
                #       of the dataset is passed as parameter.
                universe_intersection.without_feature(chosen_test),
                [test for test in budgeted_features if test != chosen_test],
                src.COSTS,
                decision_tree,
                last_added_node
            )

            # NOTE: This if assures that the feature used as root in the P(S)=1 base case is expanded only once
            if is_split_base_case and subtree.get_root_label() in universe.features:
                budgeted_features.remove(subtree.get_root_label())

            decision_tree.add_subtree(chosen_test, subtree, str(label))

        universe = universe.intersection(universe.S_star[chosen_test])
        spent += costs[chosen_test]
        universe.drop_feature(chosen_test)
        budgeted_features.remove(chosen_test)

    logger.info("End of t_A part of the procedure!")

    # If there are still some tests with cost greater than budget
    logger.info(f"Starting t_B part of the procedure")
    if len(budgeted_features) != 0 and len(universe) != 0:
        while True:
            chosen_test = pairs_maximization(universe, budgeted_features, costs)
            logger.debug("Chosen test: %s", chosen_test)

            # Set chosen_test as child of the test added in the last iteration
            backbone_label = get_backbone_label(universe, chosen_test)
            last_added_node = decision_tree.add_node(chosen_test, last_added_node, backbone_label)

            # For each label in the possible outcomes of chosen_test
            for label in eligible_labels(universe, chosen_test):
                logger.info("Label %s for test \"%s\"", label, chosen_test)
                universe_intersection = universe.intersection(universe.S_label[chosen_test][label])
                logger.debug(f"Universe intersect S[{chosen_test}][{label}]: {universe_intersection.indexes}")

                # Set the tree resulting from the recursive call as the child of chosen_test
                logger.info("t_B recursive call with test \"%s\"", chosen_test)
                subtree, is_split_base_case = build_decision_tree(
                    # NOTE: 27/02/2023 - Remove the chosen feature before the recursive call
                    #       Instead of removing it from the dataset just to add it back after the return an updated copy
                    #       of the dataset is passed as parameter.
                    universe_intersection.without_feature(chosen_test),
                    [test for test in budgeted_features if test != chosen_test],
                    src.COSTS,
                    decision_tree,
                    last_added_node
                )

                # NOTE: This if assures that the feature used as root in the P(S)=1 base case is expanded only once
                if is_split_base_case and subtree.get_root_label() in universe.features:
                    budgeted_features.remove(subtree.get_root_label())

                decision_tree.add_subtree(chosen_test, subtree, str(label))

            universe = universe.intersection(universe.S_star[chosen_test])
            spent_2 += costs[chosen_test]
            universe.drop_feature(chosen_test)
            budgeted_features.remove(chosen_test)

            # If there are no tests left, or we're running out of budget, break the loop
            if budget - spent_2 < 0 or len(budgeted_features) == 0:
                break

    logger.info("End of t_B part of the procedure!")

    # NOTE: As stated in Section 3.1 of the paper (page 15) the final recursive call is responsible for the construction
    #       of a decision tree for the objects not covered by the tests in the backbone.
    #       It's correct to say that if U is empty, this part of the procedure is skippable.
    if len(universe) != 0:
        # Set the tree resulting from the recursive call as child of the test added in the last iteration
        logger.info("Final recursive call")
        subtree, _ = build_decision_tree(universe, src.TESTS, src.COSTS, decision_tree, last_added_node)
        backbone_label = get_backbone_label(dataset, decision_tree.get_label_of_node(last_added_node))
        decision_tree.add_subtree(last_added_node, subtree, backbone_label)

    return decision_tree, False
