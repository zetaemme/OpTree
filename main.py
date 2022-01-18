from sys import argv

import pandas as pd

from cost import calculate_cost, find_budget
from dectree.dectree import DecTree
from dectree.node import LeafNode, TestNode
from dectree.test import Test
from pairs import Pairs


def main(tests_filepath: str, classes_filepath: str):
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

    pairs = Pairs(dataset)

    with open(tests_filepath, 'r', encoding='UTF-8') as f:
        raw_tests = [line.rstrip() for line in f]
        tests = [Test(test) for test in raw_tests]

    with open(classes_filepath, 'r', encoding='UTF-8') as f:
        raw_classes = [line.rstrip() for line in f]
        classes = [class_str for class_str in raw_classes]

    # FIXME: Completare la definizione della classe Cost
    test_costs = [calculate_cost(test) for test in tests]

    if pairs.number == 0:
        return DecTree(LeafNode(dataset[0]['class']))

    if pairs.number == 1:
        minimum_cost_test = min(test_costs)

        # FIXME: L'init di LeafNode è osceno
        return DecTree(
            TestNode(
                # FIXME: Dovrebbe essere il test con costro minimo che separa le classi
                str(minimum_cost_test),
                LeafNode(dataset[pairs.pair_list[0]]['class']),
                LeafNode(dataset[pairs.pair_list[1]]['class'])
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
    main(argv[1], argv[2])
