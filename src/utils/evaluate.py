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

        result_dict[separation_set_pairs.number] = separation_set

    return result_dict[max(result_dict.keys())]


def objects_kept_by_test(objects: DataFrame, test: str) -> DataFrame:
    """Computes the objects 'kept by' a given test

    Parameters
    ----------
    objects: DataFrame
        The dataset to evaluate
    test: str
        The feature that keeps the objects

    Returns
    -------
    objects_kept: A subset of objects, containing all the objects kept by test
    """
    # Computes the set difference between S and S^{*}_{test}
    sigma_test = objects[~objects.apply(tuple, 1).isin(maximum_separation_set_for_test(objects, test).apply(tuple, 1))]

    indexes = set()

    for pair in Pairs(objects).pair_list:
        if pair[0] in sigma_test.index and pair[1] in sigma_test.index:
            indexes.add(pair[0])
            indexes.add(pair[1])

    return objects.iloc[indexes]


def objects_separated_by_test(objects: DataFrame, test: str) -> DataFrame:
    """Computes the objects 'separated by' a given test

    Parameters
    ----------
    objects: DataFrame
        The dataset to evaluate
    test: str
        The feature that keeps the objects

    Returns
    -------
    objects_separated: A subset of objects, containing all the objects separated by test
    """
    maximum_separation_set = maximum_separation_set_for_test(objects, test)

    # Computes the set difference between S and S^{*}_{test}
    sigma_test = objects[~objects.apply(tuple, 1).isin(maximum_separation_set.apply(tuple, 1))]

    indexes = set()

    for pair in Pairs(objects).pair_list:
        if (pair[0] in sigma_test.index and pair[1] in maximum_separation_set.index) or \
                (pair[1] in sigma_test.index and pair[0] in maximum_separation_set.index):
            indexes.add(pair[0])
            indexes.add(pair[1])

    return objects.iloc[indexes]
