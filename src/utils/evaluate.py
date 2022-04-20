from pandas import DataFrame


def dataset_for_test(objects: DataFrame, test: str) -> dict[str, DataFrame]:
    """Calculates the S^{i}_{test} set for a given dataset

    Parameters
    ----------
    objects: The dataset to separate
    test: The feature by which we want to separate each objects in the dataset

    Returns
    -------
    dict[str, DataFrame]: A dict containing, for each distinct value for the 'test' feature, a sub-DataFrame containing
                          the objects separated by 'test'
    """
    no_duplicates_feature = objects[test].drop_duplicates()

    result = {
        key: objects[objects[test] == key]
        for key in no_duplicates_feature
    }

    return result
