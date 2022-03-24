from sys import argv

from pandas import DataFrame

from cost import calculate_cost
from src.dectree.dectree import DTOA
from src.utils import extract


def main(tests_filepath: str):
    """Inits dataset and test list in order to pass them to the algorithm"""
    dataset = DataFrame(
        data=[
            [1, 1, 2, 'A', 0.1],
            [1, 2, 1, 'A', 0.2],
            [2, 2, 1, 'B', 0.4],
            [1, 2, 2, 'C', 0.25],
            [2, 2, 2, 'C', 0.05],
        ],
        columns=['t1', 't2', 't3', 'class', 'probability']
    )

    # Reads the tests from an input file
    with open(tests_filepath, 'r', encoding='UTF-8') as f:
        test_strings = [line.rstrip() for line in f]
        tests = [extract.test_structure(test) for test in test_strings]

    # Runs the recursive algorithm that builds the optimal Decision Tree
    decision_tree = DTOA(
        objects=dataset,
        tests=tests,
        cost_fn=calculate_cost
    )


if __name__ == '__main__':
    main(argv[1])
