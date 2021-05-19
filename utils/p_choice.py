from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def p_choice(p: np.ndarray) -> int:
    """Return an index value from a probability distribution p"""
    lo: int = 0
    hi: int = p.shape[0]
    # x: float = random.random()    # same speed
    x: float = np.random.uniform(0.0, 1.0)
    cum_p: np.ndarray = np.cumsum(p)
    while lo < hi:
        mid = (lo + hi) // 2
        # Use __lt__ to match the logic in list.sort() and in heapq
        if x < cum_p[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


@njit(cache=True)
def cum_p_choice(cum_p: np.ndarray) -> int:
    """Return an index value from a probability distribution p"""
    lo: int = 0
    hi: int = cum_p.shape[0]
    # x: float = random.random()    # same speed
    x: float = np.random.uniform(0.0, 1.0)
    while lo < hi:
        mid = (lo + hi) // 2
        # Use __lt__ to match the logic in list.sort() and in heapq
        if x < cum_p[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


@njit(cache=True)
def n_choice(n: int) -> int:
    """Return a random int from 0 to n-1"""
    return np.random.randint(n)


@njit(cache=True)
def uniform() -> float:
    """Return a random float uniformly from 0.0 to 1.0"""
    return np.random.uniform(0.0, 1.0)
