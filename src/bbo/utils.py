from itertools import product
from typing import Callable

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import (
    GaussianProcessClassifier,
    GaussianProcessRegressor,
)


def construct_meshgrid(
    n_dimensions: int, grd_res: int, bounds: list[tuple] | None = None
) -> tuple:
    """Construct meshgrid.

    Args:
        n_dimensions (int): number of dimensions.
        grd_res (int): grid resolution in each dimension.
        bounds (list): list of minimum and maximum bounds for each dimension,
          defaults to 0 and 1.

    Returns:
        tuple of coordinate matrices.
    """
    if bounds is None:  # use entire range
        bounds = [(0, 1) for _ in range(n_dimensions)]

    axis_arrays = [
        np.linspace(bounds[dim][0], bounds[dim][1], grd_res) for dim in range(n_dimensions)
    ]

    return np.meshgrid(*axis_arrays, indexing="ij")


def construct_fine_meshgrid(
    x: np.ndarray,
    grd_res: int,
    limit: float,
    bounds: tuple[int, int] = (0, 1),
) -> tuple:
    """Construct fine meshgrid used when zooming in on interesting areas.

    Args:
        x (np.ndarray): point to zoom in around.
        grd_res (int): grid resolution in each dimension.
        limit (float): zoomed-in subrange in each dimension.
        bounds (tuple): maximum and minimum bounds to clip grid in each
          dimension.

    Returns:
        tuple of coordinate matrices.
    """
    axis_arrays = [
        np.linspace(x_dim - 0.5 * limit, x_dim + 0.5 * limit, grd_res) for x_dim in x
    ]
    clipped_axis_arrays = [
        np.clip(arr, bounds[0], bounds[1]) for arr in axis_arrays
    ]

    return clipped_axis_arrays


def get_farthest_point(
    x_samples: np.ndarray, n_dimensions: int, grd_res: int
) -> np.ndarray:
    """Calculate farthest grid point from corners and samples.

    Args:
        x_samples (np.ndarray): input values for samples.
        n_dimensions (int): number of dimensions.
        grd_res (int): grid resolution in each dimension.

    Returns:
        point on grid that lies farthest from corners and samples.
    """
    # Define grid points
    meshgrid = construct_meshgrid(n_dimensions, grd_res)
    ravel_meshgrid = [x.ravel() for x in meshgrid]
    grid_points = np.column_stack(ravel_meshgrid)

    # Find corners and combine coordinates with those of samples
    corners = np.array(list(product([0.0, 1.0], repeat=n_dimensions)))
    corners_x = np.vstack([corners, x_samples])

    # Find minimum distance of each grid point to a sample/corner
    dists = cdist(grid_points, corners_x)
    min_dists = dists.min(axis=1)

    # Farthest point lies at maximum of minimum distances
    max_idx = np.argmax(min_dists)
    x_next = grid_points[max_idx]

    return x_next


def grid_search(
    model: GaussianProcessRegressor,
    acq_func: Callable,
    n_dimensions: int,
    grd_res: int,
    clf_model: GaussianProcessClassifier | None = None,
    bounds: list[tuple] | None = None,
    **acq_kwargs,
) -> np.ndarray:
    """Perform grid search on progressively smaller regions until limit reached.

    Args:
        model (GaussianProcessRegressor): fitted Gaussian Process surrogate
          model.
        acq_func (Callable): acquisition function to maximise to find next
          point.
        n_dimensions (int): number of dimensions.
        grd_res (int): grid resolution in each dimension.
        clf_model (GaussianProcessClassifier): fitted Gaussian Process
          classification model if applicable.
        bounds (list): list of minimum and maximum bounds for each dimension,
          defaults to 0 and 1.
        acq_kwargs: keyword arguments for acquisition function.

    Returns:
        point on grid to investigate next.
    """
    # Get grid points for search
    meshgrid = construct_meshgrid(n_dimensions, grd_res, bounds)
    ravel_meshgrid = [x.ravel() for x in meshgrid]
    X_pred = np.column_stack(ravel_meshgrid)

    y_mean, y_std = model.predict(X_pred, return_std=True)
    Y_mean = y_mean.reshape(meshgrid[0].shape)
    Y_std = y_std.reshape(meshgrid[0].shape)

    if clf_model:  # Find probability if classification model included
        y_prob = clf_model.predict_proba(X_pred)
        Y_prob = y_prob[:, 1].reshape(meshgrid[0].shape)
        acq = acq_func(Y_prob, Y_mean, Y_std, **acq_kwargs)
    else:
        acq = acq_func(Y_mean, Y_std, **acq_kwargs)

    # Find range of longest dimension
    max_range = 1
    if bounds:
        max_range = 0
        for b in bounds:
            dim_range = b[1] - b[0]
            if max_range < dim_range:
                max_range = dim_range
    limit = max_range / (grd_res - 1)

    idx = np.argmax(acq)
    max_idx = np.unravel_index(idx, acq.shape)
    x_next = np.array([m[max_idx] for m in meshgrid])

    while True:
        meshgrid = construct_fine_meshgrid(x_next, grd_res, limit)
        ravel_meshgrid = [x.ravel() for x in meshgrid]
        X_pred = np.column_stack(ravel_meshgrid)

        y_mean, y_std = model.predict(X_pred, return_std=True)
        Y_mean = y_mean.reshape(meshgrid[0].shape)
        Y_std = y_std.reshape(meshgrid[0].shape)

        if clf_model:
            y_prob = clf_model.predict_proba(X_pred)
            Y_prob = y_prob[:, 1].reshape(meshgrid[0].shape)
            acq = acq_func(Y_mean, Y_std, Y_prob, **acq_kwargs)
        else:
            acq = acq_func(Y_mean, Y_std, **acq_kwargs)

        idx = np.argmax(acq)
        max_idx = np.unravel_index(idx, acq.shape)
        x_next = np.array([m[max_idx] for m in meshgrid])

        if limit < 1e-6:
            return x_next

        limit = limit / (grd_res - 1)


def get_circumference_points(
    x_best: np.ndarray, x_selection: np.ndarray, n_points: int = 2000
):
    """Generate a list of points along the circumference of a circle. The radius
    of the circle is the midpoint between the best point and the closest point
    from selection.

    Args:
        x_best (np.ndarray): coordinates of best point.
        x_selection (np.ndarray): coordinates of selected points to form circle.
        n_points (int): number of points to generate.

    Returns:
        list of points along circumference of circle.
    """
    # Radius of circle is midpoint between best point and closest from
    # selection
    radii = np.linalg.norm(x_selection - x_best, axis=1) / 2
    radius = np.sort(radii)[0]

    # Generate list of equidistant points along circumference
    # Use cylindrical coordinates
    points = []
    for i in range(n_points):
        phi = i * 2 * np.pi / n_points
        x0 = x_best[0] + radius * np.cos(phi)
        x1 = x_best[1] + radius * np.sin(phi)
        points.append((x0, x1))

    return points
