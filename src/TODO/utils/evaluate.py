from pandas import DataFrame, merge

from src.dataset import Pairs


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

    return intersection


def dataset_for_test(objects: DataFrame, test: str) -> dict[str, DataFrame]:
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
    separation_set = {str(key): objects.loc[objects[test] == key] for key in objects[test].unique()}
    return separation_set


def are_dataframes_equal(df1: DataFrame, df2: DataFrame) -> bool:
    """Checks if two DataFrames are equal (same cardinality and same elements)

    Parameters
    ----------
    df1: DataFrame
        The first DataFrame to compare

    df2: DataFrame
        The second DataFrame to compare

    Returns
    -------
    result: bool
        True if the two DataFrames have same cardinality and same elements, False otherwise
    """
    if len(df1) != len(df2):
        return False

    if not df1.eq(df2, axis=0).all().to_numpy().all():
        return False

    return True


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
    result = []

    for key in objects[test].unique():
        separation_set = objects.loc[objects[test] == key]
        separation_set_pairs = Pairs(separation_set)

        result.append((separation_set_pairs.number, separation_set))

    # NOTE: Returns the DataFrame associated with the maximum Pairs number
    return max(result, key=lambda x: x[0])[1]


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

    return objects.iloc[list(indexes)]


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

    return objects.iloc[list(indexes)]