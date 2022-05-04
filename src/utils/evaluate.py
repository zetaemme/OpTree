from pandas import DataFrame, merge

from src.pairs import Pairs


def dataframe_intersection(dataframes: list[DataFrame]) -> DataFrame:
    """Calculates the intersection between a given list of Pandas DataFrames

    Parameters
    ----------
    dataframes: list[DataFrame]
        The DataFrames we need to intersect

    Returns
    -------
    intersection: DataFrame
        The DataFrame resulting from the intersection
    """
    if len(dataframes) == 1:
        return dataframes[0]

    intersection = dataframes[0]

    for frame in dataframes[1:]:
        intersection = merge(intersection, frame, how='inner')

    return intersection.dropna(inplace=True)


def dataset_for_test(objects: DataFrame, test: str) -> dict[str, dict[str, DataFrame]]:
    """Calculates the S^{i}_{test} set for a given dataset

    Parameters
    ----------
    objects: DataFrame
        The dataset to separate
    test: str
        The feature by which we want to separate each objects in the dataset

    Returns
    -------
    separation_set: dict[str, DataFrame]
        A dict containing, for each distinct value for the 'test' feature, a sub-DataFrame containing the objects
        separated by 'test'
    """
    separation_set = {test: {key: objects.loc[objects[test] == key] for key in objects[test].unique()}}

    return separation_set


def maximum_separation_set_for_test(objects: DataFrame, test: str) -> DataFrame:
    """Calculates the S^{*}_{test} set for a given dataset

    Parameters
    ----------
    objects: DataFrame
        The dataset to separate
    test: str
        The feature by which we want to separate each objects in the dataset

    Returns
    -------
    maximum_separation_set: DataFrame
        A DataFrame containing the set of objects which maximizes the Pairs number for the given 'test'
    """
    result_dict = {}

    for key in objects[test].unique():
        separation_set = objects.loc[objects[test] == key]
        separation_set_pairs = Pairs(separation_set)

        result_dict[separation_set] = separation_set_pairs.number

    return max(result_dict, key=result_dict.get)
