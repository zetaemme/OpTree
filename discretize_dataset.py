from argparse import ArgumentParser
from os.path import dirname
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


def main(dataset_path: str, bins: int) -> None:
    dataset = pd.read_csv(dataset_path)
    discretizer = KBinsDiscretizer(n_bins=bins, strategy="kmeans", encode="ordinal")

    # Compute discrete columns
    continuous = dataset.select_dtypes(include="float")
    discrete_dataset_np = discretizer.fit_transform(continuous)
    discrete_dataset = pd.DataFrame(discrete_dataset_np.astype(int), columns=continuous.columns)

    # Merge new discrete columns with original discrete columns
    discrete_dataset = pd.concat([discrete_dataset, dataset.select_dtypes(exclude="float")], axis=1)

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
    parser.add_argument("-b", "--bins", type=int, help="The number of bins in which divide the continuous feature")

    args = parser.parse_args()

    main(args.filename, args.bins)
