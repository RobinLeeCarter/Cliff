from __future__ import annotations

import numpy as np
from numba import njit


# noinspection PyPep8Naming
@njit(cache=True)
def bellman_update_v(v: np.ndarray, r: np.ndarray, T: np.ndarray, gamma: float) -> np.ndarray:
    return r + gamma * np.dot(T, v)


@njit(cache=True)
def l1_norm_above(a: np.ndarray, b: np.ndarray, theta: float) -> bool:
    l1_norm: float = 0.0
    above_theta: bool = False
    for idx in np.ndindex(a.shape):
        l1_norm += abs(a[idx] - b[idx])
        if l1_norm > theta:
            above_theta: bool = True
            break
    return above_theta


# @njit(cache=True)
# def l1_norm_above_q(q1: np.ndarray, q2: np.ndarray, theta: float) -> bool:
#     l1_norm: float = 0.0
#     above_theta: bool = False
#     for idx in np.ndindex(q1.shape):
#         l1_norm += abs(q1[i, j] - q2[i, j])
#         if l1_norm > theta:
#             above_theta: bool = True
#             break
#     return above_theta
