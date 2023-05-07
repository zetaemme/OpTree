import logging
from collections import Counter
from itertools import chain, groupby
from math import fsum

import networkx as nx

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


def prune(tree: Tree, dataset: Dataset) -> Tree:
    set_depths(tree.structure, tree.root)
    compute_cutoff_metrics(tree.structure, tree.root, len(dataset))

    sorted_nodes = sorted(tree.structure.nodes, key=lambda node: tree.structure.nodes[node]["depth"])
    groups = groupby(sorted_nodes, key=lambda node: tree.structure.nodes[node]["depth"])

    scores = {
        depth: fsum(tree.structure.nodes[node]["cutoff_metric"] for node in nodes)
        for depth, nodes in groups
    }

    cutoff_depth = 0
    scores_per_depth = dict(sorted(scores.items(), key=lambda x: x[0]))
    for current_depth, current_score in scores_per_depth.items():
        successors = {key: value for key, value in scores_per_depth.items() if key > current_depth}

        if len(successors) == 1 and list(successors.values())[0] < current_score:
            cutoff_depth = list(successors.keys())[0]
            break

        for successor_score in successors.values():
            if successor_score > current_score:
                cutoff_depth = current_depth
                break

    nodes_to_remove = [
        node
        for node in tree.structure.nodes
        if tree.structure.nodes[node]["depth"] > cutoff_depth
    ]

    if len(nodes_to_remove) == 0:
        return tree

    tree.structure.remove_nodes_from(nodes_to_remove)

    for leaf in tree.leaves:
        leaf_classes = [
            class_
            for index, class_ in dataset.classes.items()
            if index in tree.structure.nodes[leaf]["objects"]
        ]

        class_counter = Counter(leaf_classes)
        tree.structure.nodes[leaf]["label"] = class_counter.most_common(1)[0][0]

    return tree
