from pandas import DataFrame


def extract_object_class(dataset: DataFrame, index: int) -> str:
    """Extracts the class label from the item in position index of a given dataset"""
    assert index >= 0, "Index should be a positive integer"
    return dataset[index]['class']
