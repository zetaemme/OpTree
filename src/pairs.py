from pandas import DataFrame


class Pairs:
    """A tuple of items having different classes

    Attributes
    ----------
    number: int
        The number of pairs for a given dataset
    pair_list: list[tuple]
        A list containing all the pairs tuples in a given dataset
    """
    def __init__(self, dataset: DataFrame) -> None:
        """Inits the number and pair_list fields"""
        # We suppose to have a 'class' column in the dataset
        assert 'class' in dataset.columns, 'Dataset should contain a \'class\' column'

        # FIXME: Scambiare con Logger
        print('CALCULATING PAIRS')

        # If item1 and item2 have a different class, they constitute a pair
        self.pair_list = [
            (i1, i2)
            for i1, d1 in dataset.iterrows()
            for i2, d2 in dataset[i1:].iterrows()
            if d1['class'] != d2['class']
        ]

        self.number = len(self.pair_list)
