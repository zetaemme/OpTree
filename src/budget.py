from src.dataset import Dataset
from src.heuristic import wolsey_greedy_heuristic
from src.separation import Separation
from src.utils import submodular_function_1


def find_budget(dataset: Dataset, separation: Separation) -> float:
    # NOTE: In the original paper alpha is marked as (1 - e^{X}), approximated with 0.35
    alpha = 0.35

    # FIXME: This should be done by Binary Search
    for budget in range(1, dataset.total_cost + 1):
        heuristc_result = wolsey_greedy_heuristic(budget, dataset, separation, submodular_function_1)

        # NOTE: Look for "test covering" meaning in paper

    return 0.0
