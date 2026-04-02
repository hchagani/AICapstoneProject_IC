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


def get_rmse(y: np.ndarray, preds: np.ndarray) -> float:
    """Calculate Root Mean Square Error (RMSE).

    Args:
        y (np.ndarray): outputs of observed data points.
        preds (np.ndarray): predicted values of observed data points from model.

    Returns:
        RMSE.
    """
    return np.sqrt(np.mean((y - preds) ** 2))


def get_baseline_mse(y: np.ndarray) -> float:
    """Calculate baseline Mean Square Error (MSE) to set minimum threshold for
    improvement.

    Args:
        y (np.ndarray): outputs of observed data points.

    Returns:
        MSE using mean of output for all predictions.
    """
    return np.mean(np.abs(y - y.mean()))
