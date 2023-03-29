from argparse import ArgumentParser
from os.path import dirname
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


def concat_onehot_cols(columns, bin_edges_):
    new_columns = []
    for idx, column in enumerate(columns):
        label = bin_edges_[idx]
        new_columns += ['{}=[{:.2f}-{:.2f}]'.format(column, label[i], label[i + 1]) for i in range(len(label) - 1)]
    return new_columns


def discretize(dataset_path: str, bins: int) -> None:
    dataset = pd.read_csv(dataset_path)
    discretizer = KBinsDiscretizer(n_bins=bins, strategy="kmeans", encode="onehot-dense")

    # Compute discrete columns
    continuous = dataset.select_dtypes(include="float")
    discrete_dataset = discretizer.fit_transform(continuous)
    new_columns = concat_onehot_cols(continuous.columns, discretizer.bin_edges_)
    discrete_dataset_df = pd.DataFrame(discrete_dataset.astype(int), columns=new_columns)

    # Merge new discrete columns with original discrete columns
    discrete = pd.concat([discrete_dataset_df, dataset.select_dtypes(exclude="float")], axis=1)

    # Saves the discrete dataset to csv
    dataset_name = dataset_path.replace("data/", "").replace(".csv", "")
    filepath = Path(dirname(__file__) + f"/data/{dataset_name}_discrete.csv")
    discrete.to_csv(filepath, index=False)


if __name__ == '__main__':
    parser = ArgumentParser(
        prog="discretize_dataset.py",
        description="Discretize the continuous columns in the dataset"
    )
    parser.add_argument("-f", "--filename", type=str, help="The CSV file containing the dataset")
    parser.add_argument("-b", "--bins", type=int, help="The number of bins in which divide the continuous feature")

    args = parser.parse_args()

    discretize(args.filename, args.bins)
