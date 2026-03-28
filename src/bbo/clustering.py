import numpy as np
from sklearn.cluster import AgglomerativeClustering


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
