from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree


def plot_correlation_matrix(
    X: np.ndarray, y: np.ndarray, figsize: tuple[int, int] = (8,6)
) -> tuple:
    """Plot correlation matrix for features X and target y.

    Args:
        X (np.ndarray): sample inputs.
        y (np.ndarray): sample outputs.
        figsize (tuple): figure size.

    Returns:
        tuple of figure and axis objects
    """
    # Combine inputs and outputs
    data = np.hstack([X, y[:, None]])

    corr_matrix = np.corrcoef(data, rowvar=False)

    fig, ax = plt.subplots(figsize=figsize)

    labels = [f"x{i}" for i in range(X.shape[1])]
    labels += ["y"]

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    ax.set_title("Correlation Matrix")
    fig.tight_layout()

    return fig, ax


def plot_1d_gp_ucb(
    x_samples: np.ndarray,
    y_samples: np.ndarray,
    x_pred: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    k: float,
    x_label: str,
    figsize: tuple[int, int] = (14, 7),
):
    """Plot prediction and standard deviation of Gaussian Process surrogate
    model with Upper Confidence Bound (UCB) acquisition function.

    Args:
        x_samples (np.ndarray): sample inputs.
        y_samples (np.ndarray): sample outputs.
        x_pred (np.ndarray): inputs used in prediction.
        y_mean (np.ndarray): predicted outputs.
        y_std (np.ndarray): uncertainty in predicted outputs.
        k (float): UCB exploration parameter.
        x_label (str): label to identify input.
        figsize (tuple): size of figure.

    Returns:
        tuple of figure and axis objects.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.scatter(
        x_samples, y_samples, edgecolor="k", s=60, alpha=0.8, label="Samples"
    )
    ax.plot(x_pred, y_mean, color="k", lw=3, label="GP mean")
    ax.fill_between(
        x_pred,
        y_mean - k * y_std,
        y_mean + k * y_std,
        color='C0',
        alpha=0.25,
        label=r"$\pm 1.96\sigma$",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel("y")
    ax.set_title(f"Predicted {x_label} from Gaussian Process Surrogate Model")

    fig.tight_layout()

    return fig, ax


def plot_2d_gp_surfaces(
    X0: np.ndarray,
    X1: np.ndarray,
    Y_mean: np.ndarray,
    Y_std: np.ndarray,
    x_samples: np.ndarray,
    y_samples: np.ndarray | None = None,
    regions: list[np.ndarray, np.ndarray, float] | None = None,
    contour: bool = True,
    x_next: np.ndarray | None = None,
    figsize: tuple[int, int] = (14, 7),
) -> tuple:
    """Plot Gaussian Process mean and standard deviation with sample points.

    Args:
        X0 (np.ndarray), X1 (np.ndarray): grid coordinates for predictions.
        Y_mean (np.ndarray): predicted mean values from Gaussian Process.
        Y_std (np.ndarray): predicted standard deviations from Gaussian Process.
        x_samples (np.ndarray): sample inputs.
        y_samples (np.ndarray): sample outputs.
        regions (list): list of regions from decision tree.
        contour (bool): plot filled contour plot rather than psuedocolour plot.
        figsize (tuple): figure size.

    Returns:
        tuple of figure and array of axes objects
    """
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    # Mean plot
    if contour:
        mean_plot = axs[0].contourf(X0, X1, Y_mean, levels=20, cmap="viridis")
    else:
        mean_plot = axs[0].pcolormesh(
            X0, X1, Y_mean, cmap="viridis", shading="nearest"
        )
    # If provided, use target values to colour sample points
    scatter_color = y_samples if y_samples is not None else "k"
    axs[0].scatter(
        x_samples[:, 0],
        x_samples[:, 1],
        c=scatter_color,
        edgecolor="k",
        s=60,
        label="Samples",
    )
    fig.colorbar(mean_plot, ax=axs[0], label="Predicted mean intensity")

    # Highlight regions if present
    if regions is not None:
        for i, region in enumerate(regions):
            x0_min, x1_min = region[0]
            x0_max, x1_max = region[1]
            width = x0_max - x0_min
            height = x1_max - x1_min
            axs[0].add_patch(
                plt.Rectangle(
                    (x0_min, x1_min),
                    width,
                    height,
                    fill=False,
                    edgecolor="red",
                    linestyle="--",
                    linewidth=2,
                    label='Region boundary' if i == 0 else None,
                )
            )

    # Plot proposed point if supplied
    if x_next is not None:
        axs[0].scatter(
            x_next[0],
            x_next[1],
            marker='X',
            s=60,
            color='r',
            label='Next point',
        )

    axs[0].legend()
    axs[0].set_xlabel("x0")
    axs[0].set_ylabel("x1")
    axs[0].set_title("Predicted mean")

    # Standard deviation plot
    if contour:
        std_plot = axs[1].contourf(X0, X1, Y_std, levels=20, cmap="magma")
    else:
        std_plot = axs[1].pcolormesh(
            X0, X1, Y_std, cmap="magma", shading="nearest"
        )
    axs[1].scatter(
        x_samples[:, 0],
        x_samples[:, 1],
        c="white",
        edgecolor="k",
        s=60,
        label="Samples",
    )
    fig.colorbar(std_plot, ax=axs[1], label="Predicted standard deviation")
    axs[1].set_xlabel("x0")
    axs[1].set_ylabel("x1")
    axs[1].set_title("Predicted standard deviation")

    fig.tight_layout()

    return fig, axs


def plot_2d_gp_acq_func_surface(
    X0: np.ndarray,
    X1: np.ndarray,
    Y_acq: np.ndarray,
    x_samples: np.ndarray,
    y_samples: np.ndarray | None = None,
    x_next: np.ndarray | None = None,
    regions: list[np.ndarray, np.ndarray, float] | None = None,
    figsize: tuple[int, int] = (8, 7),
    title: str = "Acquisition function output",
    ax: Axes = None,
) -> tuple:
    """Plot acquisition function output with sample points.

    Args:
        X0 (np.ndarray), X1 (np.ndarray): grid coordinates for predictions.
        Y_acq (np.ndarray): acquisition function output
        x_samples (np.ndarray): sample inputs.
        y_samples (np.ndarray): sample outputs.
        x_next (np.ndarray): coordinates of proposed point(s).
        regions (list): list of regions from decision tree.
        figsize (tuple): figure size.
        title (str): plot title.
        ax (Axes): axis object.

    Returns:
        tuple of figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    plot = ax.pcolormesh(X0, X1, Y_acq, cmap="viridis", shading="nearest")
    # If provided, use target values to colour sample points
    scatter_color = y_samples if y_samples is not None else "k"
    ax.scatter(
        x_samples[:, 0],
        x_samples[:, 1],
        c=scatter_color,
        edgecolor="k",
        s=60,
        label="Samples",
    )
    fig.colorbar(plot, ax=ax, label="Acquisition function output")

    # Highlight regions if present
    if regions is not None:
        for i, region in enumerate(regions):
            x0_min, x1_min = region[0]
            x0_max, x1_max = region[1]
            width = x0_max - x0_min
            height = x1_max - x1_min
            ax.add_patch(
                plt.Rectangle(
                    (x0_min, x1_min),
                    width,
                    height,
                    fill=False,
                    edgecolor="red",
                    linestyle="--",
                    linewidth=2,
                    label='Region boundary' if i == 0 else None,
                )
            )

    # Plot proposed point or candidate points if supplied
    if x_next is not None:
        x_next = np.atleast_2d(x_next)
        label = "Next point" if x_next.shape[0] == 1 else "Candidate points"
        ax.scatter(
            x_next[:, 0],
            x_next[:, 1],
            marker="X",
            s=60,
            color="r",
            label=label,
        )
    ax.legend()
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_title(title)

    fig.tight_layout()

    return fig, ax


def plot_2d_positive_negative(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    neg_x: np.ndarray,
    x_next: np.ndarray | None = None,
    figsize: tuple[int, int] = (8, 7),
):
    """Create scatter plots for positive and negative points.

    Args:
        pos_x (np.ndarray): input coordinates for positive outputs.
        pos_y (np.ndarray): array of positive outputs.
        neg_x (np.ndarray): input coordinates for negative outputs.
        x_next (np.ndarray): coordinates of proposed point.
        figsize (tuple): figure size.

    Returns:
        tuple of figure and axis objects
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    plot = ax.scatter(
        pos_x[:, 0],
        pos_x[:, 1],
        c=np.log(pos_y),
        edgecolor='k',
        s=60,
        label="Positive y"
    )
    fig.colorbar(plot, ax=ax, label="log(y)")
    ax.scatter(
        neg_x[:, 0], neg_x[:, 1], marker="x", c='b', s=60, label="Negative y"
    )
    if x_next is not None:
        ax.scatter(
            x_next[0],
            x_next[1],
            marker='x',
            s=60,
            color='r',
            label='Next point',
        )
    ax.grid()
    ax.legend()
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    fig.tight_layout()

    return fig, ax


def plot_decision_tree(
    tree: DecisionTreeRegressor,
    n_dimensions: int,
    figsize: tuple[int, int] = (14, 7),
):
    """Plot decision tree.

    Args:
        tree (DecisionTreeRegressor): decision tree regression model to plot.
        n_dimensions (int): number of dimensions.
        figsize (tuple): figure size.

    Returns:
        tuple of figure and axis objects.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    feature_names = [f"x{i}" for i in range(n_dimensions)]
    plot_tree(
        tree,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        impurity=True,
        ax=ax,
    )

    fig.tight_layout()

    return fig, ax
