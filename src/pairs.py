# from datetime import datetime
# import logging

from pandas import DataFrame

# NOTE: Change the log level here to enable DEBUG mode
# logger = logging.getLogger(__name__)

# logging.basicConfig(
#     filename=f'log/dectree_{datetime.now().strftime("%d-%m-%Y_%H.%M.%S")}.log',
#     format='%(levelname)s (%(asctime)s): %(message)s',
#     level=logging.INFO
# )


class Pairs:
    """A tuple of items having different classes"""
    
    def __init__(self, dataset: DataFrame) -> None:
        """Inits the number and pair_list fields"""
        # We suppose to have a 'class' column in the dataset
        assert 'class' in dataset.columns, 'Dataset should contain a \'class\' column'

        # logging.info(f'Calculating pairs for {dataset.shape[0]} rows...')

        # If item1 and item2 have a different class, they constitute a pair
        self.pair_list = [
            (i1, i2)
            for i1, d1 in dataset.iterrows()
            for i2, d2 in dataset[i1:].iterrows()
            if d1['class'] != d2['class']
        ]

        self.number = len(self.pair_list)

        # logging.info('Pairs calculation complete!')
