from argparse import ArgumentParser
from os.path import dirname
from pathlib import Path
from pickle import Unpickler

import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

from src.dataset import Dataset
from src.decision_tree import DecisionTree
from src.utils import prune


def get_key_from_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key

    return None


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="rocauc.py",
        description="Plots the ROC AUC curve for a given decision tree model",
    )
    parser.add_argument(
        "-dn",
        "--dataset_name",
        type=str,
        help="The Pickle file containing the model's best fit",
    )
    parser.add_argument(
        "-b",
        "--best",
        type=str,
        help="The Pickle file containing the test dataset",
    )

    args = parser.parse_args()

    if Path(dirname(__file__) + f"/model/{args.dataset_name}/{args.best}/not_pruned.pkl").is_file():
        with open(dirname(__file__) + f"/model/{args.dataset_name}/{args.best}/not_pruned.pkl", "rb") as tree_file:
            unpickler = Unpickler(tree_file)
            tree = unpickler.load()

    if Path(dirname(__file__) + f"/model/{args.dataset_name}/{args.best}/train_set.pkl").is_file():
        with open(dirname(__file__) + f"/model/{args.dataset_name}/{args.best}/train_set.pkl", "rb") as data_file:
            unpickler = Unpickler(data_file)
            train_dataset: Dataset = unpickler.load()

    aucs: dict = {}
    t_aucs = []
    for epsilon in np.arange(0.05, 1.05, 0.05):
        pruned = prune(tree, train_dataset, epsilon)
        pruned_tree = DecisionTree()
        pruned_tree.decision_tree = pruned
        pruned_tree.dataset = train_dataset

        if Path(dirname(__file__) + f"/model/{args.dataset_name}/{args.best}/test_set.pkl").is_file():
            with open(dirname(__file__) + f"/model/{args.dataset_name}/{args.best}/test_set.pkl", "rb") as data_file:
                unpickler = Unpickler(data_file)
                test_dataset: Dataset = unpickler.load()

        results = []
        predictions = []
        for row in test_dataset.data(True):
            correct = str(row[-1])
            prediction = pruned_tree.predict(row[1:-1])

            predictions.append(prediction)
            results.append(prediction == correct)

        y_true = [str(class_) for class_ in test_dataset.classes.values()]
        y_pred = predictions

        class_mapping = {
            str(class_): i for i, class_ in enumerate(set(test_dataset.classes.values()))
        }

        y_true_int = list(map(lambda x: class_mapping[x], y_true))
        y_pred_int = list(map(lambda x: -1 if x is None else class_mapping[x], y_pred))

        for idx in range(len(y_true_int)):
            if y_pred_int[idx] == -1:
                y_pred_int[idx] = 0 if y_true_int == 1 else 1

        n_classes = len(set(test_dataset.classes.values()))
        y_true_bin = label_binarize(np.array(y_true_int), classes=np.arange(n_classes))
        y_pred_bin = np.array(y_pred_int)

        if n_classes == 2:
            # Calcola la curva ROC
            fpr, tpr, _ = roc_curve(y_true_bin, y_pred_bin)

            # Calcola l'area sotto la curva ROC (AUC)
            t_aucs.append(auc(fpr, tpr))

        else:
            # Calcola la curva ROC e l'AUC per ogni classe
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            multi_auc: dict = {}
            for index in range(n_classes):
                fpr[index], tpr[index], _ = roc_curve(y_true_bin[:, index], (y_pred_bin == index).astype(int))  # type: ignore
                multi_auc[get_key_from_value(class_mapping, index)] = auc(fpr[index], tpr[index])

            aucs[epsilon] = multi_auc

    print(t_aucs)
