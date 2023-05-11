import logging
from collections import Counter
from itertools import chain
from math import fsum

import networkx as nx

from src import MAX_DEPTH
from src.dataset import Dataset
from src.tree import Tree
from src.types import Bounds, HeuristicFunction

logger = logging.getLogger("decision_tree")


def submodular_function_1(dataset: Dataset, features: list[str]) -> int:
    """Submodular function used by the heuristic

    Args:
        dataset (Dataset): The dataset on which the decision is being built
        features (list[str]): The subset of features to consider

    Returns:
        int: The difference between the dataset "Pairs" and the number of pairs in the S_star[feature] intersection
    """
    if not features:
        logger.debug(f"No features. Result = {dataset.pairs_number}")
        return dataset.pairs_number

    submodular_separation = dataset.S_star_intersection_for_features(features)

    if len(submodular_separation) < 2:
        return dataset.pairs_number

    return dataset.pairs_number - dataset.pairs_number_for(submodular_separation)


def binary_search_budget(
        dataset: Dataset,
        tests: list[str],
        costs: dict[str, float],
        search_range: Bounds,
        heuristic: HeuristicFunction,
) -> float:
    """Calculates the procedure's budget via Binary Search

    Args:
        dataset (Dataset): The dataset on which the decision is being built
        tests (list[str]): The tests for the given dataset
        costs (dict[str, float]): The costs for the tests
        search_range (list[float]): Range in which the binary search is performed
        heuristic (HeuristicFunction): Heuristic function

    Returns:
        float: The optimal budget for the procedure
    """

    # Should be (1 - e^{chi}), approximated with 0.35 in the paper
    alpha = 0.35

    budgets = [search_range.upper]
    i = 1

    while search_range.upper >= search_range.lower + 1:
        budgets.append((search_range.lower + search_range.upper) / 2)

        heuristic_result = heuristic(budgets[i], dataset, tests, costs, submodular_function_1)
        logger.debug(f"Heuristic result: {heuristic_result}")

        covered_pairs = [set(dataset.kept[test] + dataset.separated[test]) for test in heuristic_result]
        covered_pairs = set(chain(*covered_pairs))

        logger.debug(f"Pairs covered by the heuristic: {covered_pairs}")

        if len(covered_pairs) < (alpha * dataset.pairs_number):
            logger.debug("Updating upper-bound as %d", budgets[i])
            search_range.upper = budgets[i]
        else:
            logger.debug("Updating lower-bound as %d", budgets[i])
            search_range.lower = budgets[i]

        i += 1

    return budgets[i - 1]


def get_backbone_label(dataset: Dataset, feature: str) -> str:
    for key, value in dataset.S_label[feature].items():
        if value == dataset.S_star[feature]:
            return key


def set_depths(tree: nx.DiGraph, node, depth: int = 0) -> None:
    tree.nodes[node]['depth'] = depth
    for child in tree.successors(node):
        set_depths(tree, child, depth + 1)


def compute_cutoff_metrics(tree: nx.DiGraph, node, objects_number: int) -> None:
    pairs_number = tree.nodes[node]["pairs"]
    objects_in_node = len(tree.nodes[node]["objects"])

    # NOTE: Change this variable assignment to change the cutoff metric
    tree.nodes[node]["cutoff_metric"] = pairs_number * objects_in_node / objects_number

    for child in tree.successors(node):
        compute_cutoff_metrics(tree, child, objects_number)


def remove_exceeding_nodes(tree: Tree, dataset: Dataset) -> None:
    exceeding_nodes = [
        node
        for node in tree.structure.nodes
        if tree.structure.nodes[node]["depth"] > MAX_DEPTH
    ]

    if len(exceeding_nodes) > 0:
        tree.structure.remove_nodes_from(exceeding_nodes)

        for leaf in tree.leaves:
            leaf_classes = [
                class_
                for index, class_ in dataset.classes.items()
                if index in tree.structure.nodes[leaf]["objects"]
            ]

            class_counter = Counter(leaf_classes)
            tree.structure.nodes[leaf]["label"] = class_counter.most_common(1)[0][0]


def cutoff(tree: Tree, dataset: Dataset, node) -> None:
    def remove_successors(node):
        successors = list(tree.structure.successors(node))
        for successor in successors:
            remove_successors(successor)
            tree.structure.remove_node(successor)

    current_score = tree.structure.nodes[node]["cutoff_metric"]
    successors_score = fsum(tree.structure.nodes[node]["cutoff_metric"] for node in tree.structure.successors(node))

    if successors_score > current_score:
        for node_successor in tree.structure.successors(node):
            remove_successors(node_successor)

        node_classes = [
            class_
            for index, class_ in dataset.classes.items()
            if index in tree.structure.nodes[node]["objects"]
        ]

        class_counter = Counter(node_classes)
        tree.structure.nodes[node]["label"] = class_counter.most_common(1)[0][0]

        return

    for node_successor in tree.structure.successors(node):
        cutoff(tree, dataset, node_successor)


def prune(tree: Tree, dataset: Dataset) -> Tree:
    set_depths(tree.structure, tree.root)
    remove_exceeding_nodes(tree, dataset)

    compute_cutoff_metrics(tree.structure, tree.root, len(dataset))
    cutoff(tree, dataset, tree.root)

    return tree
