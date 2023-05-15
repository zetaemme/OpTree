from argparse import ArgumentParser
from os.path import dirname
from pathlib import Path
from pickle import Unpickler

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

from src.dataset import Dataset
from src.decision_tree import DecisionTree


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
        "-bf",
        "--best_fit",
        type=str,
        help="The Pickle file containing the model's best fit",
    )
    parser.add_argument(
        "-td",
        "--test_dataset",
        type=str,
        help="The Pickle file containing the test dataset",
    )

    args = parser.parse_args()

    bf_path = args.best_fit
    if Path(dirname(__file__) + f"/{bf_path}").is_file():
        with open(dirname(__file__) + f"/{bf_path}", "rb") as bf_file:
            unpickler = Unpickler(bf_file)
            bf = unpickler.load()

    decision_tree: DecisionTree = bf  # type: ignore

    td_path = args.test_dataset
    if Path(dirname(__file__) + f"/{td_path}").is_file():
        with open(dirname(__file__) + f"/{td_path}", "rb") as td_file:
            unpickler = Unpickler(td_file)
            td = unpickler.load()

    test_dataset: Dataset = td  # type: ignore

    results = []
    predictions = []
    for row in test_dataset.data(True):
        correct = str(row[-1])
        prediction = decision_tree.predict(row[1:-1])

        predictions.append(prediction)
        results.append(prediction == correct)

    y_true = [str(class_) for class_ in test_dataset.classes.values()]
    y_pred = predictions

    class_mapping = {
        class_: i for i, class_ in enumerate(set(test_dataset.classes.values()))
    }

    y_true_int = list(map(lambda x: class_mapping[x], y_true))
    y_pred_int = list(map(lambda x: -1 if x is None else class_mapping[x], y_pred))

    n_classes = len(set(test_dataset.classes.values()))
    y_true_bin = label_binarize(np.array(y_true_int), classes=np.arange(n_classes))
    y_pred_bin = np.array(y_pred_int)

    # Calcola la curva ROC e l'AUC per ogni classe
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], (y_pred_bin == i).astype(int))  # type: ignore
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()

    colors = ["red", "green", "blue"]  # Puoi personalizzare i colori delle curve

    for i in range(n_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            color=colors[i],
            lw=2,
            label='ROC curve "%s" (AUC = %.2f)'
            % (get_key_from_value(class_mapping, i), roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()
