from sys import argv

import pandas as pd

from cost import calculate_cost, find_budget
from dectree.dectree import DecTree
from dectree.node import LeafNode, TestNode
from dectree.test import Test
from pairs import Pairs
from utils import utils


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
        raw_tests = [line.rstrip() for line in f]
        tests = [Test(test) for test in raw_tests]

    test_costs = [calculate_cost(test) for test in tests]

    # Base case.
    # All objects in the dataset have the same class. A single leaf is returned.
    if pairs.number == 0:
        return DecTree(LeafNode(utils.extract_object_class(dataset, 0)))

    # Base case.
    # I have a single pair, each object in it has a different class. Two leafs are returned, having the minimum cost
    # test as root.
    if pairs.number == 1:
        # FIXME: Devo utilizzare sepcost come da definizione, non cost e basta
        min_cost_max_separability_test = min(test_costs)

        return DecTree(
            TestNode(
                str(min_cost_max_separability_test),
                LeafNode(utils.extract_object_class(dataset, 0)),
                LeafNode(utils.extract_object_class(dataset, 1))
            )
        )

    budget = find_budget(dataset, tests, classes, calculate_cost)
    spent = 0
    spent2 = 0
    # U <- S
    universe = dataset
    k = 1

    while any([test for index, test in enumerate(tests) if test_costs[index] <= budget - spent]):
        # TODO: Continuare da 'Let tk be a test...'
        pass


if __name__ == '__main__':
    main(argv[1])
