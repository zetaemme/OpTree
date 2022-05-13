from pandas import DataFrame

from cost import calculate_cost
from src.dectree.dectree import DTOA


def main() -> None:
    """Inits dataset and test list in order to pass them to the algorithm"""
    dataset = DataFrame(
        data=[
            [1, 1, 2, 'A', 0.1],
            [1, 2, 1, 'A', 0.2],
            [2, 2, 1, 'B', 0.4],
            [1, 2, 2, 'C', 0.25],
            [2, 2, 2, 'C', 0.05]
        ],
        columns=['t1', 't2', 't3', 'class', 'probability']
    )

    # Runs the recursive algorithm that builds the optimal Decision Tree
    decision_tree = DTOA(
        objects=dataset,
        tests=dataset.columns[:-2].to_list(),
        cost_fn=calculate_cost
    )

    decision_tree.show()


if __name__ == '__main__':
    main()
