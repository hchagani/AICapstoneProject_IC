import numpy as np


def get_baseline_rmse(y: np.ndarray) -> float:
    """Calculate baseline Root Mean Square Error (RMSE) to set minimum threshold
    for improvement.

    Args:
        y (np.ndarray): outputs of observed data points.

    Returns:
        RMSE using mean of output for all predictions.
    """
    return np.sqrt(np.mean((y - y.mean()) ** 2))


def get_baseline_mse(y: np.ndarray) -> float:
    """Calculate baseline Mean Square Error (MSE) to set minimum threshold for
    improvement.

    Args:
        y (np.ndarray): outputs of observed data points.

    Returns:
        MSE using mean of output for all predictions.
    """
    return np.mean(np.abs(y - y.mean()))
