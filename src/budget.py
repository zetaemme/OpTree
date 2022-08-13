from src.dataset import Dataset
from src.separation import Separation


def find_budget(dataset: Dataset, separation: Separation) -> float:
    # NOTE: In the original paper alpha is marked as (1 - e^{X}), approximated with 0.35
    alpha: float = 0.35

    return 0.0
