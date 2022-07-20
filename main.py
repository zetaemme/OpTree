from src.opsion import opsion
from pathlib import Path
from src.dataset import Dataset
from treelib import Tree


def main() -> None:
    """Inits dataset and test list in order to pass them to the algorithm"""
    path: Path = Path('data/test.csv')
    dataset: Dataset = Dataset(path)

    print(dataset.classes)

    # decision_tree: Tree = opsion(dataset)

    # joblib.dump(decision_tree, 'model/dectree.sav')
    # decision_tree.show()


if __name__ == '__main__':
    main()
