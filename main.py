# from datetime import datetime
# import logging
from sys import argv

import joblib
import pandas as pd

from src.cost import calculate_cost
from src.dectree.dectree import DTOA

# NOTE: Change the log level here to enable DEBUG mode
# logger = logging.getLogger(__name__)

# logging.basicConfig(
#     filename=f'log/dectree_{datetime.now().strftime("%d-%m-%Y_%H.%M.%S")}.log',
#     format='%(levelname)s (%(asctime)s): %(message)s',
#     level=logging.INFO
# )


def main() -> None:
    """Inits dataset and test list in order to pass them to the algorithm"""

    # logging.info(f'Loading dataset \'{argv[1]}\'')

    dataset = pd.read_csv(argv[1], names=[
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
        'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class'
    ])

    # logging.info('Dataset loaded!')

    # Runs the recursive algorithm that builds the optimal Decision Tree
    # logging.info('Starting decision tree creation...')
    decision_tree = DTOA(
        objects=dataset,
        tests=dataset.columns[:-1].to_list(),
        cost_fn=calculate_cost
    )

    # logging.info('Saving model...')
    joblib.dump(decision_tree, 'model/dectree.sav')
    # logging.info('Successfully saved model in \'model/dectree.sav\'!')

    # logging.info('Showing result...')
    decision_tree.show()


if __name__ == '__main__':
    # logging.info('Starting execution...')
    main()
    # logging.info('Execution completed successfully!')
