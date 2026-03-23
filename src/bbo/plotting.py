import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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
        ax=ax,
    )

    ax.set_title("Correlation Matrix")
    plt.tight_layout()

    return fig, ax


def plot_2d_gp_surfaces(
    X0: np.ndarray,
    X1: np.ndarray,
    Y_mean: np.ndarray,
    Y_std: np.ndarray,
    x_samples: np.ndarray,
    y_samples: np.ndarray = None,
    figsize: tuple[int, int] = (14, 7)
) -> tuple:
    """Plot Gaussian Process mean and standard deviation with sample points.

    Args:
        X0 (np.ndarray), X1 (np.ndarray): grid coordinates for predictions.
        Y_mean (np.ndarray): predicted mean values from Gaussian Process.
        Y_std (np.ndarray): predicted standard deviations from Gaussian Process.
        x_samples (np.ndarray): sample inputs.
        y_samples (np.ndarray): sample outputs.
        figsize (tuple): figure size.

    Returns:
        tuple of figure and array of axes objects
    """
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    # Mean plot
    mean_plot = axs[0].contourf(X0, X1, Y_mean, levels=20, cmap="viridis")
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
    axs[0].legend()
    axs[0].set_title("Predicted mean")

    # Standard deviation plot
    std_plot = axs[1].contourf(X0, X1, Y_std, levels=20, cmap="magma")
    axs[1].scatter(
        x_samples[:, 0],
        x_samples[:, 1],
        c="white",
        edgecolor="k",
        s=60,
        label="Samples",
    )
    fig.colorbar(std_plot, ax=axs[1], label="Predicted standard deviation")
    axs[1].set_title("Predicted standard deviation")

    plt.tight_layout()

    return fig, axs
