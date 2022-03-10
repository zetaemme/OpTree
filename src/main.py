from sys import argv

import pandas as pd

from cost import calculate_cost, find_budget
from pairs import Pairs
from src.dectree.dectree import DecTree
from src.dectree.node import LeafNode, TestNode
from src.utils import extract


def main(tests_filepath: str):
    """The main function"""
    dataset = pd.DataFrame(
        # The "dataset" at page 3 of the paper
        data=[
            [1, 1, 2, 'A', 0.1],
            [1, 2, 1, 'A', 0.2],
            [2, 2, 1, 'B', 0.4],
            [1, 2, 2, 'C', 0.25],
            [2, 2, 2, 'C', 0.05],
        ],
        columns=['t1', 't2', 't3', 'class', 'probability']
    )

    # Creates a Pairs object that holds the pairs for the given dataset
    pairs = Pairs(dataset)

    # Extracts all the class names from the dataset
    classes = {class_name for class_name in dataset[['class']]}

    # Reads the tests from an input file
    with open(tests_filepath, 'r', encoding='UTF-8') as f:
        test_strings = [line.rstrip() for line in f]
        tests = [extract.test_structure(test) for test in test_strings]

    # Inits a list with all the costs of the tests
    test_costs = [calculate_cost(test) for test in tests]

    # Inits a dictionary containing the S^{i}_{t}
    # In this case we use i (index) to obtain the ariety of the set
    items_separated_by_test = {
        test: test.evaluate_dataset_for_class(dataset, index)
        for test in tests
        for index, _ in enumerate(classes)
    }

    # Base case.
    # All objects in the dataset have the same class. A single leaf is returned.
    if pairs.number == 0:
        return DecTree(LeafNode(extract.object_class(dataset, 0)))

    # Base case.
    # I have a single pair, each object in it has a different class. Two leafs are returned, having the minimum cost
    # test as root.
    if pairs.number == 1:
        # NOTE: This set of instructions works since, in this specific case, we're working with a single pair
        #       The TestNode has been assigned to a variable in order to assign the parent node to each LeafNode
        root_node = TestNode(label=str(extract.cheapest_test(tests)))
        root_node.add_children([
            LeafNode(label=extract.object_class(dataset, 0), parent=root_node),
            LeafNode(label=extract.object_class(dataset, 1), parent=root_node)
        ])

        return DecTree(root_node)

    # Uses the FindBudget procedure to extract the correct cost budget
    budget = find_budget(dataset, tests, classes, calculate_cost)

    spent = 0
    spent2 = 0

    # U <- S
    universe = dataset

    k = 1

    # Remove from tests all tests with cost > budget
    tests = [test for test in tests if calculate_cost(test) <= budget]

    # While there's a test t with cost(t) <= budget - spent
    while any([test for index, test in enumerate(tests) if test_costs[index] <= budget - spent]):
        probability_maximizing_tests = {}

        for test in tests:
            all_objects_covered_by_test = {
                test.evaluate_dataset_for_class(dataset, index)
                for index in range(len(classes))
            }

            universe_probability = sum(universe['probability'])
            sub_universe_probability = sum(set(universe).intersection(all_objects_covered_by_test))

            probability_maximizing_tests[test] = (universe_probability - sub_universe_probability) / calculate_cost(
                test)

        if max(probability_maximizing_tests, key=probability_maximizing_tests.get) == tests[0]:
            # FIXME: Dovrebbe già esserci una struttura ad albero a cui aggiungere nodi, ma io non la ho :^)
            # TODO: "Make test[0] the root of the tree D"
            pass
        else:
            # TODO: "Make test[k] child of test t[k - 1]"
            pass


if __name__ == '__main__':
    main(argv[1])
