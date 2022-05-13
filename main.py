from sys import argv

import pandas as pd

from src.cost import calculate_cost
from src.dectree.dectree import DTOA


def main() -> None:
    """Inits dataset and test list in order to pass them to the algorithm"""
    # NOTE: The following is the init for the test dataset of the referenced paper
    # dataset = DataFrame(
    #     data=[
    #         [1, 1, 2, 'A', 0.1],
    #         [1, 2, 1, 'A', 0.2],
    #         [2, 2, 1, 'B', 0.4],
    #         [1, 2, 2, 'C', 0.25],
    #         [2, 2, 2, 'C', 0.05]
    #     ],
    #     columns=['t1', 't2', 't3', 'class', 'probability']
    # )

    # FIXME: Scambiare con Logger
    print('LOADING DATASET!')
    dataset = pd.read_csv(argv[1], names=[
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
        'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class'
    ])
    print('DONE LOADING!')
    print('\n\n')

    # Runs the recursive algorithm that builds the optimal Decision Tree
    decision_tree = DTOA(
        objects=dataset,
        tests=dataset.columns[:-1].to_list(),
        cost_fn=calculate_cost
    )

    decision_tree.show()


if __name__ == '__main__':
    main()
