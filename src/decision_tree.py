import logging
from pprint import pformat
from typing import Optional
from uuid import UUID

import src
from src.budget import find_budget
from src.dataset import Dataset
from src.extraction import cheapest_separation, eligible_labels
from src.maximization import pairs_maximization, probability_maximization
from src.tree import Tree
from src.utils import get_backbone_label

logger = logging.getLogger("decision_tree")


def build_decision_tree(dataset: Dataset, tests: list[str], costs: dict[str, float]) -> tuple[Tree, bool]:
    """Recursively builds a (log)-optimal decision tree.

    Args:
        dataset (Dataset): The dataset used to train the model
        tests (list[str]): The features from which the tree will be built
        costs (doct[str, float]): The costs for the tests

    Returns:
        Tree: The (log)-optimal decision tree
    """
    # BASE CASE: If no pairs, return a leaf labelled by the only class in dataset
    if dataset.pairs_number == 0:
        tree = Tree()

        # NOTE: Avoids insertion needless leaf
        if dataset.features and len(dataset) != 0:
            leaf = list(dataset.classes.values())[0]
            objects = dataset.indexes.tolist()

            logger.info("No pairs in dataset, setting node \"%s\" as root of the tree", leaf)
            tree.add_node(objects, None, leaf)  # type: ignore
        else:
            logger.info("No more objects in dataset")

        return tree, False

    # BASE CASE: If just one pair
    if dataset.pairs_number == 1:
        logger.info(f"Just one pair in dataset: {dataset.pairs_list[0]}")

        # Create a tree rooted by the cheapest test that separates the two items
        tree = Tree()
        split = cheapest_separation(dataset, costs, dataset.pairs_list[0])

        logger.info("Setting node \"%s\" as root of the subtree", split)
        indexes_covered = dataset.S_label_union_for(split)
        root_id = tree.add_node(indexes_covered, dataset.pairs_number_for(indexes_covered), split)

        # Add the two items as leafs labelled with the respective class
        class_1 = dataset.classes[dataset.pairs_list[0][0]]
        label_1 = str(dataset[0, dataset.features.index(split) + 1])

        class_2 = dataset.classes[dataset.pairs_list[0][1]]
        label_2 = str(dataset[1, dataset.features.index(split) + 1])

        logger.info(f"Adding leaf \"{class_1}\" as child of {tree.get_label_of_node(root_id)}")
        # FIXME: Potrebbe essere problematica la gestione dei tipi delle etichette
        tree.add_node(dataset.S_label[split][int(label_1)], None, class_1, root_id, label_1)
        logger.info(f"Adding leaf \"{class_2}\" as child of {tree.get_label_of_node(root_id)}")
        tree.add_node(dataset.S_label[split][int(label_2)], None, class_2, root_id, label_2)

        return tree, True

    budget = find_budget(dataset, tests, src.COSTS)
    logger.info("Using budget %f", budget)

    spent = 0.0
    spent_2 = 0.0

    universe = dataset.copy()

    # Removes from T all tests with cost greater than budget
    budgeted_features = [test for test in tests if costs[test] <= budget]
    logger.info(f"{len(budgeted_features)} features within budget:\n{pformat(budgeted_features)}")

    # Inits the structure
    tree = Tree()
    last_added_node: Optional[UUID] = None

    # While exists at least a test with cost equal or less than (budget - spent)
    logger.info("Starting t_A backbone construction")
    while any(cost <= budget - spent for test, cost in costs.items() if test in budgeted_features) and len(
            universe) != 0:
        chosen_test = probability_maximization(
            universe,
            [feature for feature in budgeted_features if costs[feature] <= budget - spent],
            costs
        )
        logger.debug("Test that maximizes the probability: %s", chosen_test)

        if tree.is_empty:
            # Set chosen_test as the root of the tree
            logger.info("Setting %s as root of the tree", chosen_test)
            last_added_node = tree.add_node(dataset.indexes.tolist(), dataset.pairs_number, chosen_test)  # type: ignore
        else:
            # Set chosen_test as child of the test added in the last iteration
            backbone_label = get_backbone_label(universe, chosen_test)
            logger.info(
                f"Adding node {chosen_test} as child of {tree.get_label_of_node(last_added_node)} " +
                f"with label {backbone_label}"
            )

            indexes_covered = universe.S_label_union_for(chosen_test)
            last_added_node = tree.add_node(
                indexes_covered,
                universe.pairs_number_for(indexes_covered),
                chosen_test,
                last_added_node,
                backbone_label
            )

        # For each label in the possible outcomes of chosen_test
        for label in eligible_labels(universe, chosen_test):
            logger.info("Expanding test \"%s\" with label %s", chosen_test, label)
            universe_intersection = universe.intersection(universe.S_label[chosen_test][label])
            logger.debug(f"U ∩ S[{chosen_test}][{label}]: {pformat(universe_intersection.indexes)}")

            # Set the tree resulting from the recursive call as the child of chosen_test
            logger.info("Constructing non-backbone (t_A) subtree of \"%s\"", chosen_test)
            subtree, is_split_base_case = build_decision_tree(
                # NOTE: 27/02/2023 - Remove the chosen feature before the recursive call
                #       Instead of removing it from the dataset just to add it back after the return an updated copy
                #       of the dataset is passed as parameter.
                universe_intersection.without_feature(chosen_test),
                [test for test in budgeted_features if test != chosen_test],
                src.COSTS
            )

            # NOTE: This if assures that the feature used as root in the P(S)=1 base case is not immediately re-expanded
            if is_split_base_case and subtree.get_root_label() in universe.features:
                budgeted_features.remove(subtree.get_root_label())

            tree.add_subtree(last_added_node, subtree, str(label))

        logger.debug(f"Computing U ∩ S[*][{chosen_test}]")
        universe = universe.intersection(universe.S_star[chosen_test])
        logger.debug(f"\n{pformat(universe)}")

        spent += costs[chosen_test]
        logger.debug("Adding cost of \"%s\" to spent. Total spent: %d", chosen_test, spent)

        universe.drop_feature(chosen_test)
        budgeted_features.remove(chosen_test)

    logger.info("End of t_A backbone construction!")

    # If there are still some tests with cost greater than budget
    logger.info(f"Starting t_B backbone construction")
    if len(budgeted_features) != 0 and len(universe) != 0:
        while True:
            chosen_test = pairs_maximization(universe, budgeted_features, costs)
            logger.debug("Test that maximizes the pairs number: %s", chosen_test)

            # Set chosen_test as child of the test added in the last iteration
            backbone_label = get_backbone_label(universe, chosen_test)
            logger.info(
                f"Adding node {chosen_test} as child of {tree.get_label_of_node(last_added_node)} " +
                f"with label {backbone_label}"
            )

            indexes_covered = universe.S_label_union_for(chosen_test)
            last_added_node = tree.add_node(
                indexes_covered,
                universe.pairs_number_for(indexes_covered),
                chosen_test,
                last_added_node,
                backbone_label
            )

            # For each label in the possible outcomes of chosen_test
            for label in eligible_labels(universe, chosen_test):
                logger.info("Expanding test \"%s\" with label %s", chosen_test, label)
                universe_intersection = universe.intersection(universe.S_label[chosen_test][label])
                logger.debug(f"U ∩ S[{chosen_test}][{label}]: {pformat(universe_intersection.indexes)}")

                # Set the tree resulting from the recursive call as the child of chosen_test
                logger.info("Constructing non-backbone (t_B) subtree of \"%s\"", chosen_test)
                subtree, is_split_base_case = build_decision_tree(
                    # NOTE: 27/02/2023 - Remove the chosen feature before the recursive call
                    #       Instead of removing it from the dataset just to add it back after the return an updated copy
                    #       of the dataset is passed as parameter.
                    universe_intersection.without_feature(chosen_test),
                    [test for test in budgeted_features if test != chosen_test],
                    src.COSTS
                )

                # NOTE: This if assures that the feature used as root in the P(S)=1 base case is not immediately re-expanded
                if is_split_base_case and subtree.get_root_label() in universe.features:
                    budgeted_features.remove(subtree.get_root_label())

                tree.add_subtree(last_added_node, subtree, str(label))

            logger.debug(f"Computing U ∩ S[*][{chosen_test}]")
            universe = universe.intersection(universe.S_star[chosen_test])
            logger.debug(f"\n{pformat(universe)}")

            spent_2 += costs[chosen_test]
            logger.debug("Adding cost of \"%s\" to spent2. Total spent2: %d", chosen_test, spent)

            universe.drop_feature(chosen_test)
            budgeted_features.remove(chosen_test)

            # If there are no tests left, or we're running out of budget, break the loop
            if budget - spent_2 < 0 or len(budgeted_features) == 0:
                break

    logger.info("End of t_B backbone construction!")

    # NOTE: As stated in Section 3.1 of the paper (page 15) the final recursive call is responsible for the construction
    #       of a decision tree for the objects not covered by the tests in the backbone.
    #       It's correct to say that if U is empty, this part of the procedure is skippable.
    if len(universe) != 0:
        # Set the tree resulting from the recursive call as child of the test added in the last iteration
        logger.info("Final recursive call with all the objects not in the backbone")
        subtree, _ = build_decision_tree(universe, src.TESTS, src.COSTS)
        backbone_label = get_backbone_label(dataset, tree.get_label_of_node(last_added_node))
        tree.add_subtree(last_added_node, subtree, backbone_label)

    return tree, False
