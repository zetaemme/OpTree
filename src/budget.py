import logging

from src.dataset import Dataset
from src.heuristic import wolsey_greedy_heuristic
from src.types import Bounds
from src.utils import binary_search_budget

logger = logging.getLogger(__name__)


def find_budget(dataset: Dataset) -> float:
    """Finds the optimal threshold for tests costs during decision tree creation.

    Args:
        dataset (Dataset): The dataset on which the procedure is running

    Returns:
        float: The optimal budget for the decision tree test costs
    """
    logger.info("Starting budget computation")
    search_bounds = Bounds(1.0, dataset.total_cost)
    return binary_search_budget(
        dataset, search_bounds, wolsey_greedy_heuristic
    )
