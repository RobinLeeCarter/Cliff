from __future__ import annotations

import numpy as np
from numba import njit, prange


# noinspection PyPep8Naming
@njit(cache=True)
def bellman_update_v(v: np.ndarray, r: np.ndarray, T: np.ndarray, gamma: float) -> np.ndarray:
    return r + gamma * np.dot(T, v)


@njit(cache=True)
def l1_norm_above(a: np.ndarray, b: np.ndarray, theta: float) -> bool:
    l1_norm: float = 0.0
    above_theta: bool = False
    # noinspection PyTypeChecker
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

@njit(cache=True, parallel=True)
def derive_v_from_q(policy_matrix: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    v[s] = Σa π(a|s) . q(s, a)
    :returns np.einsum('ij,ij->i', policy_matrix, q_matrix) where policy_matrix is non-zero, to avoid q = -inf
    """
    out = np.zeros(shape=policy_matrix.shape[0], dtype=np.float64)
    for i in prange(policy_matrix.shape[0]):
        for j in range(policy_matrix.shape[1]):
            if policy_matrix[i, j] != 0.0:  # also side-stepping q[i, j] = -inf
                out[i] += policy_matrix[i, j] * q[i, j]
    return out
