import numpy as np
from scipy.stats import norm


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


def clf_ucb(
    prob: np.ndarray, mean: np.ndarray, std: np.ndarray, k: float = 1.0
):
    """Calculate product of classifier probability and Upper Confidence Bound
    (UCB).

    Args:
        prob (np.ndarray): classifier probabilities.
        mean (np.ndarray): predicted mean values.
        std (np.ndarray): standard deviations from the predicted mean.
        k (float): exploration parameter.

    Returns:
        numpy array of product of classifier probability and UCB values for
          each point.
    """
    return prob * ucb(mean, std, k=k)


def prob_improv(mean: np.ndarray, std: np.ndarray, y_max: float):
    """Calculate Probability of Improvement (PI).

    Args:
        mean (np.ndarray): predicted mean values.
        std (np.ndarray): standard deviations from the predicted mean.
        y_max (float): maximum output value.

    Returns:
        numpy array of PI values for each point.
    """
    z = (mean - y_max) / (std + 1e-12)

    return norm.cdf(z)


def clf_prob_improv(
    prob: np.ndarray, mean: np.ndarray, std: np.ndarray, y_max: float
):
    """Calculate product of classifier probability and Probability of
    Improvement (PI).

    Args:
        prob (np.ndarray): classifier probabiities.
        mean (np.ndarray): predicted mean values.
        std (np.ndarray): standard deviations from the predicted mean.
        y_max (float): maximum output value.

    Returns:
        numpy array of product of classifier probability and PI values for each
          point.
    """
    return prob * prob_improv(mean, std, y_max)
