import numpy as np
from pathlib import Path

# List of number of initial data points in order of function ID
N_POINTS = [10, 10, 15, 30, 20, 20, 30, 40]
FINAL_WEEK = 14  # After all queries have been processed


def get_data_dir() -> Path:
    """Get data directory.

    Returns:
        Path object to data directory.
    """
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent.parent
    data_dir = root_dir / "data"

    return data_dir


def load_data(function_id: int, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Load data from inputs and outputs files for particular function ID, and
    curtail according to 

    Args:
        function_id (int): function ID from 1 to 8.
        n_points (int): length of data arrays to return.

    Returns:
        tuple of input and output numpy arrays.
    """
    data_dir = get_data_dir()

    inputs = np.load(f"{data_dir}/function_{function_id}_inputs.npy")[:n_points]
    outputs = np.load(
        f"{data_dir}/function_{function_id}_outputs.npy"
    )[:n_points]

    return inputs, outputs


def get_current_weeks_points(
    function_id: int, week: int
) -> tuple[np.ndarray, np.ndarray]:
    """Get known data points for current week.

    Args:
        function_id (int): function ID from 1 to 8.
        week (int): current week number.

    Returns:
        number of known data points for current week.
    """
    # Data is in chronological order
    n_points = N_POINTS[function_id - 1] + week - 1

    return load_data(function_id, n_points)


def get_final_maximum(function_id: int) -> tuple[np.ndarray, float, float]:
    """Get coordinates and output for data point that yields maximum output
    after all queries have been submitted (i.e. for the final result). Compare
    this point with the maximum output from the initial data set.

    Args:
        function_id (int): function ID from 1 to 8.

    Returns:
        tuple of coordinates of best point, output from best point, and
          difference between outputs from best point and that from initial data
          set.
    """
    # Get indices of best points from final and initial data sets
    X, y = get_current_weeks_points(function_id, week=FINAL_WEEK)
    final_max_idx = np.argmax(y)
    initial_max_idx = np.argmax(y[:N_POINTS[function_id - 1]])

    final_max_input = X[final_max_idx]
    final_max_output = y[final_max_idx]
    initial_max_output = y[initial_max_idx]

    return (
        final_max_input,
        final_max_output,
        final_max_output - initial_max_output,
    )
