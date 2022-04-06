from __future__ import annotations
import random

import numpy as np
from numba import njit


@njit(cache=True)
def _set_seeds(numba_np_random_seed: int, numba_random_seed: int):
    np.random.seed(numba_np_random_seed)
    random.seed(numba_random_seed)


@njit(cache=True)
def p_choice(p: np.ndarray) -> int:
    """
    :param p: probability distribution
    :return: a drawn index value from p
    """
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
    """
    :param cum_p: cumulative probability distribution
    :return: a drawn index value from p
    """
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
def uniform_choice_from_int_array(arr: np.ndarray) -> int:
    """Return a random value from a numpy int array"""
    size: int = arr.size
    if size == 1:
        return arr[0]
    else:
        index = np.random.randint(size)
        return arr[index]


@njit(cache=True)
def choose_argmax_index(values: np.ndarray) -> int:
    """Return a random argmax index from a numpy float array"""
    max_val: float = np.max(values)
    max_indices: np.ndarray = np.flatnonzero(values == max_val)
    index: int = uniform_choice_from_int_array(max_indices)
    return index


@njit(cache=True)
def uniform(low: float = 0.0, high: float = 1.0) -> float:
    """Return a random float uniformly from low to high"""
    return np.random.uniform(low, high)


@njit(cache=True)
def unit_uniform() -> float:
    """Return a random float uniformly from 0.0 to 1.0 (as fast as possible)"""
    return np.random.uniform(0.0, 1.0)
