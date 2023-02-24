from argparse import ArgumentParser
from os.path import dirname
from pathlib import Path

import numpy as np
import pandas as pd


def main(dataset_path: str) -> None:
    dataset = pd.read_csv(dataset_path)

    discrete_dataset = pd.DataFrame()
    for feature in dataset.select_dtypes(include='float').columns:
        # Select the optimal bins to tripartite the dataset's column
        sorted_column = dataset[feature].sort_values(kind='mergesort')
        # NOTE: We use the 0.00000000001 subtraction to avoid unlabeled values
        bins = [split.min() - 0.00000000001 for split in np.array_split(sorted_column, 3)[1:]]
        bins.insert(0, sorted_column.min() - 0.00000000001)
        bins.append(float("inf"))

        # Creates column tri-partition and labels the values
        discrete_dataset[feature] = pd.cut(
            x=dataset[feature].astype(float),
            bins=bins,
            labels=["Low", "Medium", "High"]
        )
        dataset.drop(labels=feature, axis=1, inplace=True)

    # Adds back the discrete features
    for feature in dataset.columns:
        discrete_dataset[feature] = dataset[feature]

    # Saves the discrete dataset to csv
    dataset_name = dataset_path.replace("data/", "").replace(".csv", "")
    filepath = Path(dirname(__file__) + f"/data/{dataset_name}_discrete.csv")
    discrete_dataset.to_csv(filepath, index=False)


if __name__ == '__main__':
    parser = ArgumentParser(
        prog="discretize_dataset.py",
        description="Discretize the continuous columns in the dataset"
    )
    parser.add_argument("-f", "--filename", type=str, help="The CSV file containing the dataset")

    args = parser.parse_args()

    main(args.filename)
