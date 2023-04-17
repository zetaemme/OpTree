import logging

from src.dataset import Dataset
from src.heuristic import wolsey_greedy_heuristic
from src.types import Bounds
from src.utils import binary_search_budget

logger = logging.getLogger(__name__)


def find_budget(
        dataset: Dataset,
        tests: list[str],
        costs: dict[str, float]
) -> float:
    """Finds the optimal threshold for tests costs during decision tree creation.

    Args:
        dataset (Dataset): The dataset on which the procedure is running
        tests (list[str]): The tests for the given dataset
        costs (dict[str, float]): The costs for the tests

    Returns:
        float: The optimal budget for the decision tree test costs
    """
    logger.info(f"Starting budget computation | Lower bound: 1.0 | Upper bound: {dataset.total_cost}")
    search_bounds = Bounds(1.0, dataset.total_cost)
    return binary_search_budget(
        dataset, tests, costs, search_bounds, wolsey_greedy_heuristic
    )
