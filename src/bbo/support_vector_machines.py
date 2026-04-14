import numpy as np
from sklearn import svm


def get_svm_classifier(
    X: np.ndarray, y: np.ndarray, threshold: float | None = None
):
    """Create Support Vector Machine classifier model and fit to data. Data is
    split into high and low classes, where the boundary is defined by the
    threshold. If no threshold is supplied, the boundary is the mean output.
    Uses Radial Basis Function (RBF) kernel.

    Args:
        X (np.ndarray): inputs of observed data points.
        y (np.ndarray): outputs of observed data points.
        threshold (float): output threshold for boundary.

    Returns:
        tuple of fitted Support Vector Classification model and vector of
          labels indicating whether data point lies above boundary
    """
    if threshold is None:
        # Define high-low value boundary at mean
        threshold = np.mean(y)

    labels = (y > threshold).astype(int)

    clf = svm.SVC(kernel="rbf", C=1, gamma="scale")
    clf.fit(X, labels)

    return clf, labels
