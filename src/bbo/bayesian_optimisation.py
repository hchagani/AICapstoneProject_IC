from sklearn.gaussian_process import (
    GaussianProcessClassifier,
    GaussianProcessRegressor,
)
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from bbo.enums import KernelType
from bbo.random import create_rng_seed


def get_clf_model(
    n_dimensions: int,
    seed_input: str,
    initial_length_scale: float = 0.1,
    length_scale_bounds: tuple[float] = (1e-2, 100),
    nu: float = 1.5,
) -> GaussianProcessClassifier:
    """Get Gaussian Process classification model.

    Args:
        n_dimensions (int): number of input dimensions.
        seed_input (str): string used to generate seed.
        initial_length_scale (float): initial guess at length scale for all
          dimensions.
        length_scale_bounds (tuple): lower and upper bounds for length scale.
        nu (float): smoothness parameter for Matern kernel.

    Returns:
        Gaussian Process classifier model.
    """
    kernel = ConstantKernel(1.0, (1e-2, 100))
    kernel *= Matern(
        length_scale=[initial_length_scale] * n_dimensions,
        length_scale_bounds=length_scale_bounds,
        nu=nu,
    )

    model = GaussianProcessClassifier(
        kernel=kernel,
        n_restarts_optimizer=10,
        max_iter_predict=100,
        random_state=create_rng_seed(seed_input),
    )

    return model


def get_reg_model(
    n_dimensions: int,
    seed_input: str,
    kernel_type: KernelType = KernelType.RBF,
    initial_length_scale: float = 0.1,
    length_scale_bounds: tuple[float, float] = (1e-2, 100),
    nu: float = 1.5,
    **model_kwargs,
) -> GaussianProcessRegressor | None:
    """Get Gaussian Process regression model.

    Args:
        n_dimensions (int): number of input dimensions.
        seed_input (str): string used to generate seed.
        kernel_type (KernelType): either RBF or Matern.
        initial_length_scale (float): initial guess at length scale for all
          dimensions.
        length_scale_bounds (tuple): lower and upper bounds for length scale.
        nu (float): smoothness parameter for Matern kernel.
        model_kwargs: keyword arguments for model.

    Returns:
        Gaussian Process regression model.
    """
    kernel = None
    if kernel_type == KernelType.RBF:
        kernel = kernel_type.value(
            length_scale=[initial_length_scale] * n_dimensions,
            length_scale_bounds=length_scale_bounds,
        )

    if kernel_type == KernelType.MATERN:
        kernel = kernel_type.value(
            length_scale=[initial_length_scale] * n_dimensions,
            length_scale_bounds=length_scale_bounds,
            nu=nu,
        )

    model = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        normalize_y=True,
        random_state=create_rng_seed(seed_input),
        **model_kwargs,
    )

    return model
