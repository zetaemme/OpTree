from src.dataset import Dataset
from src.heuristic import wolsey_greedy_heuristic
from src.separation import Separation
from src.utils import submodular_function_1


def find_budget(dataset: Dataset, separation: Separation) -> float:
    """Finds the optimal threshold for tests costs during decision tree creation.

    Args:
        dataset (Dataset): The dataset on which the procedure is running
        separation (Separation): The class containing the S* and S^i sets

    Returns:
        float: The optimal budget for the decision tree test costs
    """
    # Should be (1 - e^{chi}), approximated with 0.35
    alpha = 0.35

    # FIXME: This should be done by Binary Search
    for budget in range(1, dataset.total_cost + 1):
        heuristic_result = wolsey_greedy_heuristic(
            budget,
            dataset,
            separation,
            submodular_function_1
        )

        covered_pairs = list(
            set(separation.kept[test] + separation.separated[test])
            for test in heuristic_result
        )

        if len(covered_pairs) >= (alpha * dataset.pairs_number):
            return budget
