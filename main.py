from pathlib import Path

from src.dataset import Dataset
from src.separation import Separation

# from src.decision_tree import build_decision_tree
# from treelib import Tree

DEBUG = False

if DEBUG:
    import sys
    import timeit


def main() -> None:
    """Inits dataset and test list in order to pass them to the algorithm"""
    path: Path = Path('data/test.csv')
    dataset: Dataset = Dataset(path)

    separation: Separation = Separation(dataset)

    # decision_tree: Tree = build_decision_tree(dataset, separation)

    # joblib.dump(decision_tree, 'model/dectree.sav')
    # decision_tree.show()


if __name__ == '__main__':
    if DEBUG:
        result = timeit.timeit(stmt='main()', globals=globals(), number=10)
        print(f'{result / 10}')
        sys.exit(0)

    main()
