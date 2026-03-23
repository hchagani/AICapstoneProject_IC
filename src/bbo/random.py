import hashlib


def create_rng_seed(seed_input: str) -> int:
    """Create seed for random number generator from input string.

    Args:
        seed_input (str): string used to generate seed.

    Returns:
        seed for random number generator.
    """
    hash_bytes = hashlib.sha256(seed_input.encode("utf-8")).digest()[:8]

    return int.from_bytes(hash_bytes, "big") % (2 ** 32)
