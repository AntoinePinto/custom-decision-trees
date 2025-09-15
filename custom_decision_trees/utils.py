import math
import random
from typing import Tuple


def bold(text: str) -> str:
    """
    Apply ANSI bold formatting to a given string.

    Parameters
    ----------
    text : str
        The input string to format.

    Returns
    -------
    str
        The input string wrapped with ANSI bold formatting codes.
    """

    return "\033[1m" + text + "\033[0m"

def get_node_coordinates(
        id: int,
        max_depth: int,
    ) -> Tuple[float, float]:
    """
    Compute normalized (x, y) coordinates for a tree node in a binary tree layout.

    Parameters
    ----------
    id : int
        The node ID in a binary tree (0-based index).
    max_depth : int
        The maximum depth of the tree for normalization.

    Returns
    -------
    tuple[float, float]
        The (x, y) coordinates of the node, where:
        - x ∈ [0, 1] represents horizontal position,
        - y ∈ [0, 1] represents vertical position (top = 1, bottom = 0).
    """

    depth = math.floor(math.log2(id + 1))
    x = (1 + 2*(id + 1 - 2**depth)) / (2**(depth + 1))
    y = (max_depth - depth) / max_depth

    return x, y

def get_random_cardinalities(
        nb_options: int,
        max_cardinality: int = 2,
        size: int = 50000,
    ):
    """
    Generate a random sample of cardinalities based on combinatorial weights.

    Each cardinality `k` in the range [1, max_cardinality] is assigned a weight
    proportional to the number of possible subsets of size `k` from `nb_options`.
    The function then randomly samples `size` values of `k` according to these
    weighted probabilities.

    Parameters
    ----------
    nb_options : int
        The total number of available options from which subsets can be formed.
    max_cardinality : int, optional
        The maximum subset size to consider (default is 2).
    size : int, optional
        The number of random samples to generate (default is 50,000).

    Returns
    -------
    list[int]
        A list of sampled cardinalities, each value between 1 and `max_cardinality`.
    """

    k_options = list(range(1, max_cardinality + 1))
    weights = [math.comb(nb_options, k) for k in k_options]
    ks = random.choices(k_options, weights=weights, k=size) # noqa: S311

    return ks
