import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering


def get_agglomerative_clusterer(n_dimensions: int):
    """Get agglomerative (bottom-up hierarchical) clusterer. Sets a distance
    threshold at or above which clusters will not be merged.

    Args:
        n_dimensions (int): number of dimensions.

    Returns:
        agglomerative clusterer object.
    """
    typical_dist = np.sqrt(n_dimensions / 6)  # typical Euclidean distance
    distance_cutoff = typical_dist / 2  # do not merge clusters above this distance

    return AgglomerativeClustering(
        linkage="complete",  # use maximum distance between points
        metric="euclidean",
        distance_threshold=distance_cutoff,
        n_clusters=None,
    )


def get_dbscan_labels(
    points: np.ndarray, max_distance: float, min_samples: int
):
    """Use Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
    algorithm to find clusters and return labels.

    Args:
        points (np.ndarray): array of data points.
        max_distance (float): maximum distance between two points for them to
          be considered to be in each other's neighbourhood.
        min_samples (int): minimum number of points in a neighbourhood.

    Returns:
        list of labels corresponding to clusters for each point.
    """
    clusterer = DBSCAN(eps=max_distance, min_samples=min_samples).fit(points)

    return clusterer.labels_
