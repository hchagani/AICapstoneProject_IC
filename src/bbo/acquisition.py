import numpy as np


def ucb(mean: np.ndarray, std: np.ndarray, k: float = 1.0):
    """Calculate Upper Confidence Bound (UCB).

    Args:
        mean (np.ndarray): predicted mean values.
        std (np.ndarray): standard deviations from the predicted mean.
        k (float): exploration parameter.

    Returns:
        numpy array of UCB values for each point.
    """
    return mean + k * std
