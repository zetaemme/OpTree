import logging
from pathlib import Path
from pickle import HIGHEST_PROTOCOL, Unpickler, dump
from pprint import pformat
from typing import Optional, Self
from uuid import UUID

import numpy as np

import src
from src.budget import find_budget
from src.dataset import Dataset
from src.extraction import cheapest_separation, eligible_labels
from src.maximization import pairs_maximization, probability_maximization
from src.tree import Tree
from src.utils import get_backbone_label, prune

logger = logging.getLogger("decision_tree")


class DecisionTree:
    dataset: Optional[Dataset] = None
    decision_tree: Optional[Tree] = None

    def _build_decision_tree(self, dataset: Dataset, tests: list[str], costs: dict[str, float]) -> tuple[Tree, bool]:
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
                leaf = str(list(dataset.classes.values())[0])
                objects = dataset.indexes.tolist()

                logger.info("No pairs in dataset, setting node \"%s\" as root of the tree", leaf)
                tree.add_node(objects, 0, leaf)  # type: ignore
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
            class_1 = str(dataset.classes[dataset.pairs_list[0][0]])
            label_1 = str(dataset[0, dataset.features.index(split) + 1])

            class_2 = str(dataset.classes[dataset.pairs_list[0][1]])
            label_2 = str(dataset[1, dataset.features.index(split) + 1])

            logger.info(f"Adding leaf \"{class_1}\" as child of {tree.get_label_of_node(root_id)}")
            tree.add_node(dataset.S_label[split][label_1], 0, class_1, root_id, label_1)
            logger.info(f"Adding leaf \"{class_2}\" as child of {tree.get_label_of_node(root_id)}")
            tree.add_node(dataset.S_label[split][label_2], 0, class_2, root_id, label_2)

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
        backbone_label = ""

        # While exists at least a test with cost equal or less than (budget - spent)
        logger.info("Starting t_A construction")
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
                last_added_node = tree.add_node(
                    dataset.indexes.tolist(),  # type: ignore
                    dataset.pairs_number,
                    chosen_test
                )
            else:
                # Set chosen_test as child of the test added in the last iteration
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
                subtree, is_split_base_case = self._build_decision_tree(
                    # NOTE: 27/02/2023 - Remove the chosen feature before the recursive call
                    #       Instead of removing it from the dataset just to add it back after the return an updated copy
                    #       of the dataset is passed as parameter.
                    universe_intersection,
                    [test for test in budgeted_features if test != chosen_test],
                    src.COSTS
                )

                tree.add_subtree(last_added_node, subtree, str(label))

            # NOTE: 09/05/2023 - Given any node in the tree, this is how it's children are created
            #                                            t_{k - 1}
            #                                            /   |    \
            #                                          A     B ... Z
            #                                        /       |      \
            #                                      C1       C2      C3
            #                    where:
            #                        * A = S^1_{t_{k - 1}} -> Non-backbone
            #                        * B = S^2_{t_{k - 1}} -> Non-backbone
            #                        * Z = S^*_{t_{k - 1}} -> Backbone
            #                   In the previous versions of the code we wrongly chose Z = S^*_{t_k} instead, resulting
            #                   in a poor labeling of the backbone.
            #                   The error was never discovered since we worked with binary features only.
            #                   By choosing the backbone edge label here we avoid the update of chosen_test, resulting
            #                   in the previous error.
            backbone_label = get_backbone_label(universe, chosen_test)

            logger.debug(f"Computing U ∩ S[*][{chosen_test}]")
            universe = universe.intersection(universe.S_star[chosen_test])
            logger.debug(f"\n{pformat(universe)}")

            spent += costs[chosen_test]
            logger.debug("Adding cost of \"%s\" to spent. Total spent: %d", chosen_test, spent)

            budgeted_features.remove(chosen_test)

        logger.info("End of t_A construction!")

        # If there are still some tests with cost greater than budget
        logger.info(f"Starting t_B construction")
        if len(budgeted_features) != 0 and len(universe) != 0:
            while True:
                chosen_test = pairs_maximization(universe, budgeted_features, costs)
                logger.debug("Test that maximizes the pairs number: %s", chosen_test)

                # Set chosen_test as child of the test added in the last iteration
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
                    subtree, is_split_base_case = self._build_decision_tree(
                        # NOTE: 27/02/2023 - Remove the chosen feature before the recursive call
                        #       Instead of removing it from the dataset just to add it back after the return an updated
                        #       copy of the dataset is passed as parameter.
                        universe_intersection,
                        [test for test in budgeted_features if test != chosen_test],
                        src.COSTS
                    )

                    tree.add_subtree(last_added_node, subtree, str(label))

                # NOTE: 09/05/2023 - Given any node in the tree, this is how it's children are created
                #                                            t_{k - 1}
                #                                            /   |    \
                #                                          A     B ... Z
                #                                        /       |      \
                #                                      C1       C2      C3
                #                    where:
                #                        * A = S^1_{t_{k - 1}} -> Non-backbone
                #                        * B = S^2_{t_{k - 1}} -> Non-backbone
                #                        * Z = S^*_{t_{k - 1}} -> Backbone
                #                   In the previous versions of the code we wrongly chose Z = S^*_{t_k} instead,
                #                   resulting in a poor labeling of the backbone.
                #                   The error was never discovered since we worked with binary features only.
                #                   By choosing the backbone edge label here we avoid the update of chosen_test,
                #                   resulting in the previous error.
                backbone_label = get_backbone_label(universe, chosen_test)

                logger.debug(f"Computing U ∩ S[*][{chosen_test}]")
                universe = universe.intersection(universe.S_star[chosen_test])
                logger.debug(f"\n{pformat(universe)}")

                spent_2 += costs[chosen_test]
                logger.debug("Adding cost of \"%s\" to spent2. Total spent2: %d", chosen_test, spent)

                budgeted_features.remove(chosen_test)

                # If there are no tests left, or we're running out of budget, break the loop
                if budget - spent_2 < 0 or len(budgeted_features) == 0:
                    break

        logger.info("End of t_B construction!")

        # NOTE: As stated in Section 3.1 of the paper (page 15) the final recursive call is responsible for the
        #       construction of a decision tree for the objects not covered by the tests in the backbone.
        #       It's correct to say that if U is empty, this part of the procedure is skippable.
        if len(universe) != 0:
            # Set the tree resulting from the recursive call as child of the test added in the last iteration
            logger.info("Final recursive call with all the objects not in the backbone")
            subtree, _ = self._build_decision_tree(universe, src.TESTS, src.COSTS)
            backbone_label = get_backbone_label(dataset, tree.get_label_of_node(last_added_node))
            tree.add_subtree(last_added_node, subtree, backbone_label)

        return tree, False

    def fit(self, dataset: Dataset, tests: list[str], costs: dict[str, float], dataset_name: str) -> None:
        self.dataset = dataset
        decision_tree, _ = self._build_decision_tree(dataset, tests, costs)

        assert decision_tree.check_leaves_objects(dataset.classes), "The decision tree is not correct!"

        logger.info("Pruning resulting tree")
        decision_tree = prune(decision_tree, dataset)

        logger.info("End of procedure!")

        with open(f"model/decision_tree_{dataset_name}.pkl", "wb") as pickle_file:
            dump(decision_tree, pickle_file, HIGHEST_PROTOCOL)

        self.decision_tree = decision_tree

    @classmethod
    def from_pickle(cls, path: str, dataset: Dataset) -> Self:
        if Path(path).is_file():
            with open(path, "rb") as model_file:
                logger.info("Loading model from Pickle file")
                unpickler = Unpickler(model_file)
                decision_tree = unpickler.load()

                model = cls()
                model.decision_tree = decision_tree
                model.dataset = dataset

                return model

    def predict(self, obj: np.ndarray) -> str:
        if self.decision_tree is None:
            raise RuntimeError("It is mandatory to call the \"fit\" method before \"predict\"!")

        def predict_for_node(current_node) -> str:
            current_node_label = self.decision_tree.get_label_of_node(current_node)

            successors = list(self.decision_tree.structure.successors(current_node))
            if len(successors) == 0:
                return str(current_node_label)

            obj_value_for_feature = str(obj[self.dataset.features.index(current_node_label)])

            for successor in self.decision_tree.structure.successors(current_node):
                edge_label = self.decision_tree.structure.get_edge_data(current_node, successor).get("label")
                if obj_value_for_feature == edge_label:
                    return predict_for_node(successor)

        return predict_for_node(self.decision_tree.root)

    def print(self) -> None:
        self.decision_tree.print()
