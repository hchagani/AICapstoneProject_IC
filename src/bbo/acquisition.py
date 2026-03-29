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
    std = np.maximum(std, 1e-12)  # avoid division by zero error
    z = (mean - y_max) / std

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


def expect_improv(
    mean: np.ndarray, std: np.ndarray, y_max: float, xi: float = 0.05
):
    """Calculate Expected Improvement (EI).

    Args:
        mean (np.ndarray): predicted mean values.
        std (np.ndarray): standard deviations from the predicted mean.
        y_max (float): maximum output value.
        xi (float): exploration parameter.

    Returns:
        numpy array of EI values for each point.
    """
    std = np.maximum(std, 1e-12)  # avoid division by zero error
    improvement = mean - y_max - xi
    z = improvement / std

    return improvement * norm.cdf(z) + std * norm.pdf(z)


def get_acquisition_functions(y_max: float, k: float = 1.96, xi: float = 0.05):
    """Get list of acquisition functions with exploration parameter arguments
    if applicable. This is used in loops.

    Args:
        y_max (float): maximum output value.
        k (float): exploration parameter for Upper Confidence Bound (UCB).
        xi (float): exploration parameter for Expected Improvement (EI).

    Returns:
        list of acqusition functions ready to use in loops.
    """
    return [
        {
            "name": f"Upper Confidence Bound (k = {k})",
            "acq_kwargs": {
                "acq_func": ucb,
                "k": k,
            },
        },
        {
            "name": "Probability of Improvement",
            "acq_kwargs": {
                "acq_func": prob_improv,
                "y_max": y_max,
            },
        },
        {
            "name": f"Expected Improvement (xi = {xi})",
            "acq_kwargs": {
                "acq_func": expect_improv,
                "xi": xi,
                "y_max": y_max,
            },
        },
    ]
