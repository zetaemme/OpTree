from random import randint
from typing import Callable

from pandas import DataFrame, Series
from treelib import Tree

from src.cost import find_budget
from src.pairs import Pairs
from src.utils import evaluate, extract

last_added_node = None


        if next_test_node is not None:
            self.last_added_node = self.tree.get_node(next_test_node)

    def add_subtree(self, sub_tree: 'DecTree') -> None:
        """Appends a given sub_tree to the currently last_added_node

        Parameters
        ----------
        sub_tree: DecTree
            The subtree to append
        """
        # FIXME: Ci sono 2 nodi 't1'?
        self.tree.paste(self.last_added_node, sub_tree.tree)
        self.last_added_node = sub_tree.last_added_node


def DTOA(objects: DataFrame, tests: list[str], cost_fn: Callable[[Series], int]) -> DecTree:
    """Recursive function that creates an optimal Decision Tree

    Parameters
    ----------
    objects: DataFrame
        The dataset containing the objects to classify
    tests: list[str]
        The test to use in order to classify the objects of the dataset
    cost_fn: Callable[[Series], int]
        A function returning the effective cost of a given test

    Returns
    -------
    decision_tree: DecTree
        An optimal Decision Tree
    """

    # Creates a Pairs object that holds the pairs for the given dataset
    pairs = Pairs(objects)

    # Extracts all the class names from the dataset
    classes = {ariety: class_name for ariety, class_name in enumerate(set(objects['class']))}

    # Inits a dictionary containing the S^{i}_{test} for each feature in tests
    items_separated_by_test = {
        test: evaluate.dataset_for_test(objects, test)
        for test in tests
    }

    # Base case.
    # All objects in the dataset have the same class. A single leaf is returned.
    if pairs.number == 0:
        return DecTree(Node(extract.object_class(objects, 0), node_type=NodeType.LeafNode))

    # Base case.
    # I have a single pair, each object in it has a different class. Two leafs are returned, having the minimum cost
    # test as root.
    if pairs.number == 1:
        # NOTE: This set of instructions works since, in this specific case, we're working with a single pair.
        #       The TestNode has been assigned to a variable in order to assign the parent node to each LeafNode
        root_node = Node(label=extract.cheapest_test(objects, tests, cost_fn), node_type=NodeType.TestNode)
        decision_tree = DecTree(root_node)

        decision_tree.add_children([
            Node(label=extract.object_class(objects, 0), node_type=NodeType.LeafNode),
            Node(label=extract.object_class(objects, 1), node_type=NodeType.LeafNode)
        ])

        return decision_tree

    # Uses the FindBudget procedure to extract the correct cost budget
    budget = find_budget(objects, tests, set(classes.values()), cost_fn, pairs.number)

    spent = 0
    spent2 = 0

    # U <- S
    # NOTE: The U variable is called universe to remark the parallelism of this problem with the Set Cover problem
    universe = objects

    k = 1

    # Remove from tests all tests with cost > budget
    tests = [test for test in tests if cost_fn(objects[test]) <= budget]

    # Builds an empty decision tree, the starting point of the recursive procedure
    decision_tree = DecTree()

    # While there's a test t with cost(t) <= budget - spent
    while any([test for test in tests if cost_fn(objects[test]) <= budget - spent]):
        # NOTE: Since we need to extract the test t_{k} which maximizes the function:
        #           (probability(universe) - probability(universe intersect items_separated_by_t_{k}))/cost(t_{k})
        #       we can simply create a list containing all tests which cost is less than budget - spent.
        #       Then we can use the cheapest possible test, since it maximizes the function in the majority of times.

        # FIXME: Si dovrebbe implementare in modo che venga fatto un check per costi uguali, dato che in quel caso va
        #        massimizzato il numeratore
        tests_eligible_for_maximization = extract.tests_costing_less_than(objects, tests, cost_fn, budget - spent)

        # NOTE: Corresponds to t_k
        probability_maximizing_test = extract.cheapest_test(objects, tests_eligible_for_maximization, cost_fn)

        if probability_maximizing_test == tests[0]:
            # Make test[0] the root of the tree D
            decision_tree.add_root(Node(probability_maximizing_test, node_type=NodeType.TestNode))
        else:
            # Make test[k] child of test t[k - 1]
            decision_tree.add_children(Node(probability_maximizing_test, node_type=NodeType.TestNode))

        # Extracts S^{*}_{t_k}
        maximum_separated_class_from_tk = extract.maximum_separated_class(items_separated_by_test,
                                                                          probability_maximizing_test,
                                                                          objects[probability_maximizing_test].unique())

        # For each i in {1...l}
        for value in objects[probability_maximizing_test].unique():
            items_separated_by_tk = items_separated_by_test[probability_maximizing_test][str(value)]


            resulting_intersection = evaluate.dataframe_intersection([items_separated_by_tk, universe])

            # If U intersect S^{i}_{t_k} is not empty and S^{i}_{t_k} != S^{*}_{t_k}
            if not resulting_intersection.empty and \
                    not evaluate.are_dataframes_equal(items_separated_by_tk, maximum_separated_class_from_tk):
                # Make D^{i} the recursive call to DTOA, called on resulting_intersection
                decision_tree.add_subtree(DTOA(resulting_intersection, tests, cost_fn))

        # NOTE: The warning can be ignored since resulting_intersection is granted to be assigned during the for loop
        universe = resulting_intersection

        spent += cost_fn(objects[probability_maximizing_test])
        tests.remove(probability_maximizing_test)
        k += 1

    if tests:
        while budget - spent2 >= 0 or tests:
            # NOTE: Since we need to extract the test t_{k} which maximizes the function:
            #           (pairs(universe) - pairs(universe intersect items_separated_by_t_{k}))/cost(t_{k})
            #       we can simply use the cheapest possible test,
            #       since it maximizes the function in the majority of times.

            # FIXME: Si dovrebbe implementare in modo che venga fatto un check per costi uguali, dato che in quel caso
            #        va massimizzato il numeratore
            pairs_maximizing_test = extract.cheapest_test(objects, tests, cost_fn)

            # Set t_{k} as child of t_{k - 1}
            decision_tree.add_children(Node(pairs_maximizing_test, node_type=NodeType.TestNode))

            # Extracts S^{*}_{t_k}
            maximum_separated_class_from_tk = extract.maximum_separated_class(items_separated_by_test,
                                                                              pairs_maximizing_test,
                                                                              objects[pairs_maximizing_test].unique())

            # For each i in {1...l}
            for value in objects[pairs_maximizing_test].unique():
                items_separated_by_tk = items_separated_by_test[pairs_maximizing_test][str(value)]

                resulting_intersection = evaluate.dataframe_intersection([items_separated_by_tk, universe])

                # If U intersect S^{i}_{t_k} is not empty and S^{i}_{t_k} != S^{*}_{t_k}
                if not resulting_intersection.empty and \
                        not evaluate.are_dataframes_equal(items_separated_by_tk, maximum_separated_class_from_tk):
                    # Make D^{i} the recursive call to DTOA, called on resulting_intersection
                    decision_tree.add_subtree(DTOA(resulting_intersection, tests, cost_fn))

            universe = resulting_intersection

            spent2 += cost_fn(objects[pairs_maximizing_test])
            tests.remove(pairs_maximizing_test)
            k += 1

    # Create a new decision tree to be added as child of decision_tree, created with a recursive call to DTOA
    decision_tree_prime = DTOA(universe, tests, cost_fn)
    decision_tree.add_subtree(decision_tree_prime)

    return decision_tree
