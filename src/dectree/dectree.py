from dataclasses import dataclass, field
from typing import Callable, Sequence, Union

from pandas import DataFrame, Series, merge

from src.cost import find_budget
from src.dectree.node import LeafNode, TestNode
from src.pairs import Pairs
from src.utils import evaluate, extract


@dataclass
class DecTree:
    """Represents a Decision Tree

    Attributes
    ----------
    root: Represents the root of the Decision Tree. Can be empty.
    last_added_node: The last added node of the Decision Tree
    """
    root: Union[LeafNode, TestNode, None]
    last_added_node: Union[LeafNode, TestNode] = field(init=False)

    def __post_init__(self) -> None:
        if self.root is None:
            return

        if not self.root.children:
            self.last_added_node = self.root
        else:
            current = self.root

            while current.children:
                current = current.children[-1]

            # NOTE: We always pick the rightmost child as last_added_node.
            #       This is right most of the time, and it's also as I intended it from the referenced paper.
            self.last_added_node = current

    def add_children(self, children: Union[Union[LeafNode, TestNode], Sequence[Union[LeafNode, TestNode]]]) -> None:
        """Adds a children to the last added node of this tree

        Parameters
        ----------
        children: A single child or the list of children to add as children of this node
        """
        self.last_added_node.add_children(children)

        if isinstance(children, Sequence):
            self.last_added_node = children[-1]
        else:
            self.last_added_node = children

    def add_root(self, new_root: Union[LeafNode, TestNode]) -> None:
        """Adds a node as root, if the root is None.

        Parameters
        ----------
        new_root: The new root of the tree

        Raises
        ------
        ValueError: This operation will block the execution if invoked on a non-empty Decision Tree
        """
        assert self.root is None, ValueError('Root is not None!')
        self.root = new_root

    def add_subtree(self, subtree: 'DecTree') -> None:
        """Adds the DecTree subtree as a child of the last added node

        Args:
            subtree (DecTree): The tree to be added as subtree
        """
        subtree.root.parent = self.last_added_node
        self.add_children(subtree.root)
        self.last_added_node = subtree.last_added_node

    @classmethod
    def build_empty_tree(cls) -> 'DecTree':
        """API to handle the creation of an empty Decision Tree.
        Needed in order to grant the existence of a tree to start from to add nodes.
        """
        return DecTree(None)


def DTOA(objects: DataFrame, tests: list[str], cost_fn: Callable[[Series], int]) -> DecTree:
    """Recursive function that creates an optimal Decision Tree

    Parameters
    ----------
    objects: The dataset containing the objects to classify
    tests: The test to use in order to classify the objects of the dataset
    cost_fn: A function returning the effective cost of a given test

    Returns
    -------
    DecTree: An optimal Decision Tree
    """

    # Creates a Pairs object that holds the pairs for the given dataset
    pairs = Pairs(objects)

    # Extracts all the class names from the dataset
    classes = {ariety: class_name for ariety, class_name in enumerate(set(objects['class']))}

    # Inits a dictionary containing the S^{i}_{t}
    # In this case we use i (index) to obtain the ariety of the set
    items_separated_by_test = {
        # FIXME: How can I calculate the S^{i}_{t} sets?
        test: DataFrame(test.evaluate_dataset_for_class(objects, index))
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
        root_node = TestNode(label=extract.cheapest_test(objects, tests, cost_fn))
        decision_tree = DecTree(root_node)

        decision_tree.add_children([
            LeafNode(label=extract.object_class(objects, 0), parent=root_node),
            LeafNode(label=extract.object_class(objects, 1), parent=root_node)
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
    decision_tree = DecTree.build_empty_tree()

    # While there's a test t with cost(t) <= budget - spent
    while any([test for test in tests if cost_fn(objects[test]) <= budget - spent]):
        # NOTE: Since we need to extract the test t_{k} which maximizes the function:
        #           (probability(universe) - probability(universe intersect items_separated_by_t_{k}))/cost(t_{k})
        #       we can simply create a list containing all tests which cost is less than budget - spent.
        #       Then we can use the cheapest possible test, since it always maximizes the function (?).
        tests_eligible_for_maximization = extract.tests_costing_less_than(objects, tests, cost_fn, budget - spent)

        # NOTE: Corresponds to t_k
        probability_maximizing_test = extract.cheapest_test(objects, tests_eligible_for_maximization, cost_fn)

        if probability_maximizing_test == tests[0]:
            # Make test[0] the root of the tree D
            decision_tree.add_root(TestNode(probability_maximizing_test))
        else:
            # Make test[k] child of test t[k - 1]
            decision_tree.add_children(TestNode(probability_maximizing_test, parent=decision_tree.last_added_node))

        # Extracts S^{*}_{t_k}
        maximum_separated_class_from_tk = extract.maximum_separated_class(
            items_separated_by_test,
            probability_maximizing_test,
            set(classes.values())
        )

        # For each i in {1...l}
        for class_label in classes.values():
            items_separated_by_tk = DataFrame(
                data=set(items_separated_by_test[probability_maximizing_test][class_label]),
                columns=objects.columns
            )

            resulting_intersection = merge(items_separated_by_tk, universe, how='inner')

            # If U intersect S^{i}_{t_k} is not empty and S^{i}_{t_k} != S^{*}_{t_k}
            if resulting_intersection and items_separated_by_tk != maximum_separated_class_from_tk:
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
            #       we can simply use the cheapest possible test, since it always maximizes the function (?).
            pairs_maximizing_test = extract.cheapest_test(objects, tests, cost_fn)

            # Set t_{k} as child of t_{k - 1}
            decision_tree.add_children(TestNode(str(pairs_maximizing_test), parent=decision_tree.last_added_node))

            # Extracts S^{*}_{t_k}
            maximum_separated_class_from_tk = extract.maximum_separated_class(
                items_separated_by_test,
                pairs_maximizing_test,
                set(classes.values())
            )

            # For each i in {1...l}
            for class_label in classes.values():
                items_separated_by_tk = DataFrame(
                    data=set(items_separated_by_test[pairs_maximizing_test][class_label]),
                    columns=objects.columns
                )

                resulting_intersection = merge(items_separated_by_tk, universe, how='inner')

                # If U intersect S^{i}_{t_k} is not empty and S^{i}_{t_k} != S^{*}_{t_k}
                if resulting_intersection and items_separated_by_tk != maximum_separated_class_from_tk:
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
