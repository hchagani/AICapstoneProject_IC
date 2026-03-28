import hashlib

import numpy as np
from scipy.stats import qmc


def create_rng_seed(seed_input: str) -> int:
    """Create seed for random number generator from input string.

    Args:
        seed_input (str): string used to generate seed.

    Returns:
        seed for random number generator.
    """
    hash_bytes = hashlib.sha256(seed_input.encode("utf-8")).digest()[:8]

    return int.from_bytes(hash_bytes, "big") % (2 ** 32)


def latin_hypercube(n_samples: int, n_dimensions: int, seed_input: str):
    """Generate samples using Latin Hypercube Sampling (LHS).

    Args:
        n_samples (int): number of samples to generate.
        n_dimensions (int): number of dimensions.
        seed_input (str): string used to generate random seed.

    Returns:
        coordinates of generated samples.
    """
    sampler = qmc.LatinHypercube(
        d=n_dimensions, seed=create_rng_seed(seed_input)
    )

    return sampler.random(n_samples)


def sample_regions(
    n_samples: int,
    regions: list[tuple[np.ndarray, np.ndarray, float]],
    seed_input: str,
    temperature: float = 0.7,
):
    """Generate random candidates (samples) in regions. The number of random
    samples generated in each region is proportional to the region's value. In
    other words, more candidates will be generated in promising regions as
    identified by a decision tree model.

    Args:
        n_samples (int): number of samples to generate.
        regions (list): list of regions.
        seed_input (str): string used to generate seed.
        temperature (float): sharpness parameter for softmax distribution.
          Lower values concentrate samples in more promising regions.

    Returns:
        coordinates of candidates and their corresponding regions.
    """
    n_dimensions = len(regions[0][0])

    values = np.array([r[2] for r in regions])

    # Softmax weighting
    scaled = values / temperature
    scaled -= np.max(scaled)
    weights = np.exp(scaled)
    weights /= weights.sum()

    # Allocate samples
    n_candidates_per_region = np.floor(weights * n_samples).astype(int)
    # If needed increase samples in most promising region to get to requested
    # amount
    n_candidates_per_region[np.argmax(weights)] += (
        n_samples - n_candidates_per_region.sum()
    )

    samples = []
    region_ids = []
    for i, ((lower, upper, _), n) in enumerate(
        zip(regions, n_candidates_per_region)
    ):
        if n != 0:  # there are candidates in the region
            rng = np.random.default_rng(seed=create_rng_seed(seed_input))
            points = lower + (upper - lower) * rng.random((n, n_dimensions))
            samples.append(points)
            region_ids.extend([i] * n)

    return np.vstack(samples), np.array(region_ids)
