import logging
import os
from sys import argv

import joblib
import pandas as pd

from src.cost import calculate_cost
from src.dectree.dectree import DTOA

if os.path.exists('dectree.log'):
    os.remove('dectree.log')

# NOTE: Change the log level here to enable DEBUG mode
logger = logging.getLogger(__name__)

logging.basicConfig(
    filename='dectree.log',
    format='%(levelname)s (%(asctime)s): %(message)s',
    level=logging.INFO
)


def main() -> None:
    """Inits dataset and test list in order to pass them to the algorithm"""
    # NOTE: The following is the init for the test dataset of the referenced paper
    # dataset = pd.DataFrame(
    #     data=[
    #         [1, 1, 2, 'A', 0.1],
    #         [1, 2, 1, 'A', 0.2],
    #         [2, 2, 1, 'B', 0.4],
    #         [1, 2, 2, 'C', 0.25],
    #         [2, 2, 2, 'C', 0.05]
    #     ],
    #     columns=['t1', 't2', 't3', 'class', 'probability']
    # )

    logging.info(f'Loading dataset \'{argv[1]}\'')

    dataset = pd.read_csv(argv[1], names=[
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
        'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class'
    ])

    logging.info('Dataset loaded!')

    # Runs the recursive algorithm that builds the optimal Decision Tree
    logging.info('Starting decision tree creation...')
    decision_tree = DTOA(
        objects=dataset,
        tests=dataset.columns[:-1].to_list(),
        cost_fn=calculate_cost
    )

    logging.info('Saving model...')
    joblib.dump(decision_tree, 'model/dectree.sav')

    logging.info('Showing result...')
    decision_tree.show()


if __name__ == '__main__':
    logging.info('Starting execution...')
    main()
    logging.info('Execution completed successfully!')
