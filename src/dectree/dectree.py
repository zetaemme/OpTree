from dataclasses import dataclass, field
from typing import Callable, Sequence, Union

from pandas import DataFrame

from src.cost import find_budget
from src.dectree.node import LeafNode, TestNode
from src.dectree.test import Test
from src.pairs import Pairs
from src.utils import extract


@dataclass
class DecTree:
    """Represents a Decision Tree"""
    root: Union[LeafNode, TestNode, None]
    last_added_node: Union[LeafNode, TestNode] = field(init=False)

    def __post_init__(self) -> None:
        if self.root is None:
            return

        if not self.root.children:
            self.last_added_node = self.root
        else:
            # TODO: Assign last_added_node to the last added value in the children of root
            pass

    def add_children(self, children: Union[Union[LeafNode, TestNode], Sequence[Union[LeafNode, TestNode]]]) -> None:
        """Adds a children to the last added node of this tree"""
        self.last_added_node.add_children(children)

        if isinstance(children, Sequence):
            # NOTE: Since, in this branch, we're assuming you can add multiple children to the last_added_node,
            #       we use the effectively last added child to update the last_added_node value
            self.last_added_node = children[-1]
        else:
            self.last_added_node = children

    def add_root(self, new_root: Union[LeafNode, TestNode]) -> None:
        """
        Adds a node as root, if the root is None.
        Throws a ValueError otherwise.
        """
        assert self.root is None, ValueError('Root is not None!')
        self.root = new_root
        self.last_added_node = self.root

    def add_subtree(self, subtree: 'DecTree') -> None:
        """Adds the DecTree subtree as a child of the last added node"""
        subtree.root.parent = self.last_added_node
        self.add_children(subtree.root)
        self.last_added_node = subtree.last_added_node

    @classmethod
    def build_empty_tree(cls) -> 'DecTree':
        """
        API to handle the creation of an empty Decision Tree.
        Needed in order to grant the existence of a tree to start from to add nodes.
        """
        return DecTree(None)


def DTOA(objects: DataFrame, tests: list[Test], cost_fn: Callable[[Test], int]) -> DecTree:
    """Recursive function that creates an optimal Decision Tree"""

    # Creates a Pairs object that holds the pairs for the given dataset
    pairs = Pairs(objects)

    # Inits a list with all the costs of the tests
    test_costs = [cost_fn(test) for test in tests]

    # Extracts all the class names from the dataset
    classes = {class_name for class_name in objects[['class']]}

    # Inits a dictionary containing the S^{i}_{t}
    # In this case we use i (index) to obtain the ariety of the set
    items_separated_by_test = {
        test: test.evaluate_dataset_for_class(objects, index)
        for test in tests
        for index, _ in enumerate(classes)
    }

    # Base case.
    # All objects in the dataset have the same class. A single leaf is returned.
    if pairs.number == 0:
        return DecTree(LeafNode(extract.object_class(objects, 0)))

    # Base case.
    # I have a single pair, each object in it has a different class. Two leafs are returned, having the minimum cost
    # test as root.
    if pairs.number == 1:
        # NOTE: This set of instructions works since, in this specific case, we're working with a single pair.
        #       The TestNode has been assigned to a variable in order to assign the parent node to each LeafNode
        root_node = TestNode(label=str(extract.cheapest_test(tests)))
        decision_tree = DecTree(root_node)

        decision_tree.add_children([
            LeafNode(label=extract.object_class(objects, 0), parent=root_node),
            LeafNode(label=extract.object_class(objects, 1), parent=root_node)
        ])

        return decision_tree

    # Uses the FindBudget procedure to extract the correct cost budget
    budget = find_budget(objects, tests, classes, cost_fn)

    spent = 0
    spent2 = 0

    # U <- S
    # NOTE: The U variable is called universe to remark the parallelism of this problem with the Set Cover problem
    universe = objects

    k = 1

    # Remove from tests all tests with cost > budget
    tests = [test for test in tests if cost_fn(test) <= budget]

    # Builds an empty decision tree, the starting point of the recursive procedure
    decision_tree = DecTree.build_empty_tree()

    # While there's a test t with cost(t) <= budget - spent
    while any([test for index, test in enumerate(tests) if test_costs[index] <= budget - spent]):
        # NOTE: Since we need to extract the test t_{k} which maximizes the function:
        #           (probability(universe) - probability(universe intersect items_separated_by_t_{k}))/cost(t_{k})
        #       we can simply create a list containing all tests which cost is less than budget - spent.
        #       Then we can use the cheapest possible test, since it always maximizes the function (?).
        tests_eligible_for_maximization = extract.tests_costing_less_than(tests, budget - spent)
        probability_maximizing_test = extract.cheapest_test(tests_eligible_for_maximization)

        if probability_maximizing_test == tests[0]:
            # Make test[0] the root of the tree D
            decision_tree.add_root(TestNode(str(probability_maximizing_test)))
        else:
            # Make test[k] child of test t[k - 1]
            decision_tree.add_children(TestNode(str(probability_maximizing_test), parent=decision_tree.last_added_node))

        # FIXME: Duplicated code, this 'for loop' should be converted to a function
        for class_label in classes:
            # FIXME: Handle all types of collection as slices of Dataset type.
            #        Doing so, it results much easier to prototype the calls to the algorithm.
            items_separated_by_tk = set(items_separated_by_test[probability_maximizing_test][class_label])

            # NOTE: Corresponds to S^{*}_{t_k}
            maximum_separated_class_from_tk = max(items_separated_by_test[probability_maximizing_test])

            resulting_intersection = items_separated_by_tk.intersection(set(universe.as_data_frame()))

            if resulting_intersection and items_separated_by_tk != maximum_separated_class_from_tk:
                decision_tree.add_subtree(DTOA(resulting_intersection, tests, cost_fn))

        universe = resulting_intersection

        spent += cost_fn(probability_maximizing_test)
        tests.remove(probability_maximizing_test)
        k += 1

    if tests:
        while True:
            # NOTE: Since we need to extract the test t_{k} which maximizes the function:
            #           (pairs(universe) - pairs(universe intersect items_separated_by_t_{k}))/cost(t_{k})
            #       we can simply use the cheapest possible test, since it always maximizes the function (?).
            pairs_maximizing_test = extract.cheapest_test(tests)

            # Set t_{k} as child of t_{k - 1}
            decision_tree.add_children(TestNode(str(pairs_maximizing_test), parent=decision_tree.last_added_node))

            # FIXME: Duplicated code, this 'for loop' should be converted to a function
            for class_label in classes:
                items_separated_by_tk = set(items_separated_by_test[pairs_maximizing_test][class_label])

                # NOTE: Corresponds to S^{*}_{t_k}
                maximum_separated_class_from_tk = max(items_separated_by_test[pairs_maximizing_test])

                resulting_intersection = items_separated_by_tk.intersection(set(universe.as_data_frame()))

                if resulting_intersection and items_separated_by_tk != maximum_separated_class_from_tk:
                    decision_tree.add_subtree(DTOA(resulting_intersection, tests, cost_fn))

            universe = resulting_intersection

            spent2 += cost_fn(pairs_maximizing_test)
            tests.remove(pairs_maximizing_test)
            k += 1

            # Repeat-until exit condition
            if budget - spent2 < 0 or not tests:
                break

    decision_tree_prime = DTOA(universe, tests, cost_fn)
    decision_tree.add_subtree(decision_tree_prime)

    return decision_tree
