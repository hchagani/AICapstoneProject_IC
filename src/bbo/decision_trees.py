import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from bbo.random import create_rng_seed


def get_decision_tree(max_depth: int, min_samples_leaf: int, seed_input: str):
    """Create decision tree.

    Args:
        max_depth (int): maximum depth of tree.
        min_samples_leaf (int): minimum samples in a leaf node.
        seed_input (str): string used to generate seed.

    Returns:
        decision tree regression model.
    """
    return DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=create_rng_seed(seed_input),
    )


def get_regions(tree: DecisionTreeRegressor, n_dimensions: int):
    """Extract regions from decision tree model.

    Args:
        tree (DecisionTreeRegressor): decision tree model.
        n_dimensions (int): number of dimensions.

    Returns:
        list of ranges and values for regions in the format:
          [([lower_bounds_per_dimension,...], [upper_bounds_per_dimension,...], value)]
    """
    regions = []

    def set_region_values_and_ranges(
        tree: DecisionTreeRegressor,
        node: int,
        lower: np.ndarray,
        upper: np.ndarray,
        regions: list[tuple[np.ndarray, np.ndarray, float]]
    ):
        """Traverse decision tree and extract ranges and values of leaf nodes.

        Args:
            tree (DecisionTreeRegressor): decision tree model.
            node (int): decision tree node.
            lower (np.ndarray): array of lower bounds in each dimension.
            upper (np.ndarray): array of upper bounds in each dimension.
            regions (list): list of ranges and values for regions.

        Returns:
            list of ranges and values for regions.
        """
        feature = tree.tree_.feature[node]
        if feature == -2:  # leaf node
            value = tree.tree_.value[node][0, 0]
            regions.append((lower.copy(), upper.copy(), value))
            return

        # Explore left and right branches if decision node
        threshold = tree.tree_.threshold[node]

        upper_left = upper.copy()
        upper_left[feature] = min(upper_left[feature], threshold)
        set_region_values_and_ranges(
            tree, tree.tree_.children_left[node], lower, upper_left, regions
        )

        lower_right = lower.copy()
        lower_right[feature] = max(lower_right[feature], threshold)
        set_region_values_and_ranges(
            tree, tree.tree_.children_right[node], lower_right, upper, regions
        )

    # Start at head node
    set_region_values_and_ranges(
        tree, 0, np.zeros(n_dimensions), np.ones(n_dimensions), regions
    )

    return regions


def get_random_forests_model(
    n_estimators: int, max_depth: int, min_samples_leaf: int, seed_input: str
):
    """Create random forests ensemble model.

    Each tree in the ensemble model is formed by taking a random subset of
    features, finding the best possible split for each selected feature by
    minimising the mean squared error, and selecting the best split among
    these features. This is repeated until either further splits would result
    in a leaf having fewer samples than the allowable minimum or the maximum
    depth of the tree has been reached.

    Args:
        n_estimators (int): number of trees in the forest.
        max_depth (int): maximum depth of each tree.
        seed_input (str): string used to generate seed.

    Returns:
        ramdom forests ensemble model.
    """
    return RandomForestRegressor(
        bootstrap=True,  # sample with replacement
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state = create_rng_seed(seed_input),
    )


def get_extra_trees_model(
    n_estimators: int, max_depth: int, min_samples_leaf: int, seed_input: str
):
    """Create extremely randomised trees ensemble model.

    Each tree in the ensemble model is formed by taking a random subset of
    features, picking a random split threshold for each feature, and selecting
    the best split among those random splits. This is repeated until either
    further splits would result in a leaf having fewer samples than the
    allowable minimum or the maximum depth of the tree has been reached.

    Args:
        n_estimators (int): number of trees in the forest.
        max_depth (int): maximum depth of each tree.
        seed_input (str): string used to generate seed.

    Returns:
        extremely randomised trees ensemble model.
    """
    return ExtraTreesRegressor(
        bootstrap=True,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state = create_rng_seed(seed_input),
    )


def get_ensemble_stats(
    model: RandomForestRegressor | ExtraTreesRegressor, X_pred: np.ndarray
):
    """Get predictions for grid coordinates from each tree that forms an
    ensemble model. Calculate the means and standard deviations at each point
    on the grid.

    Args:
        model (RandomForestRegressor | ExtraTreesRegressor): ensemble model.
        X_pred (np.ndarray): grid coordinates for predictions.

    Returns:
        tuple of numpy arrays of means and standard deviations at grid points.
    """
    predictions = np.stack([t.predict(X_pred) for t in model.estimators_])

    return predictions.mean(axis=0), predictions.std(axis=0)
