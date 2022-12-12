import logging

from src.dataset import Dataset
from src.heuristic import wolsey_greedy_heuristic
from src.separation import Separation
from src.utils import binary_search_budget

logger = logging.getLogger(__name__)


def find_budget(dataset: Dataset, separation: Separation) -> float:
    """Finds the optimal threshold for tests costs during decision tree creation.

    Args:
        dataset (Dataset): The dataset on which the procedure is running
        separation (Separation): The class containing the S* and S^i sets

    Returns:
        float: The optimal budget for the decision tree test costs
    """
    logger.info("Starting budget computation")
    return binary_search_budget(
        dataset, separation, [1.0, dataset.total_cost], wolsey_greedy_heuristic
    )
