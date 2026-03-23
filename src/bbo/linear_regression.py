import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from bbo.random import create_rng_seed


def get_lr_models() -> dict:
    """Create linear regression models.

    Returns:
        dictionary of linear regression models.
    """
    models = {
        "linear": Pipeline(
            [
                ("poly", PolynomialFeatures(degree=1, include_bias=False)),
                ("lr", LinearRegression()),
            ]
        ),
        "quadratic": Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("lr", LinearRegression()),
            ]
        ),
        "cubic": Pipeline(
            [
                ("poly", PolynomialFeatures(degree=3, include_bias=False)),
                ("lr", LinearRegression()),
            ]
        ),
    }

    return models


def loocv(X: np.ndarray, y: np.ndarray) -> dict:
    """Perform Leave One Out Cross Validation on linear regression models. Uses
    Root Mean Square Error (RMSE) and MSE as evaluation metrics.

    Args:
        X (np.ndarray): array of inputs for observed data points.
        y (np.ndarray): array of outputs for observed data points.

    Returns:
        dictionary of RMSE, RMSE spread and RMSE relative to output for each
          model.
    """
    stats = {}
    for name, model in get_lr_models().items():
        scores = cross_val_score(
            model, X, y, cv=LeaveOneOut(), scoring="neg_mean_squared_error"
        )
        mse_per_fold = -scores
        mse = mse_per_fold.mean()
        mse_std = np.std(mse_per_fold)
        rmse = np.sqrt(mse)
        rmse_std = np.std(np.sqrt(mse_per_fold))
        stats[name] = {
            "MSE": mse,
            "MSE spread": mse_std,
            "RMSE": rmse,
            "RMSE spread": rmse_std,
            "relative RMSE": rmse / (y.max() - y.min())
        }

    return stats


def kfoldcv(
    X: np.ndarray, y: np.ndarray, seed_input: str, n_splits: int = 10
) -> dict:
    """Perform K-Fold Cross Validation on linear regression models. Uses Root
    Mean Square Error (RMSE) and MSE as evaluation metrics.

    Args:
        X (np.ndarray): array of inputs for observed data points.
        y (np.ndarray): array of outputs for observed data points.
        seed_input (str): string used to generate random seed.
        n_splits (int): number of folds.

    Returns:
        dictionary of RMSE, RMSE spread and RMSE relative to output for each
          model.
    """
    stats = {}
    random_seed = create_rng_seed(seed_input)
    for name, model in get_lr_models().items():
        scores = cross_val_score(
            model,
            X,
            y,
            cv=KFold(n_splits=n_splits, shuffle=True, random_state=random_seed),
            scoring="neg_mean_squared_error",
        )
        mse_per_fold = -scores
        mse = mse_per_fold.mean()
        mse_std = np.std(mse_per_fold)
        rmse = np.sqrt(mse)
        rmse_std = np.std(np.sqrt(np.sqrt(mse_per_fold)))
        stats[name] = {
            "MSE": mse,
            "MSE spread": mse_std,
            "RMSE": rmse,
            "RMSE spread": rmse_std,
            "relative RMSE": rmse / (y.max() - y.min())
        }

    return stats
