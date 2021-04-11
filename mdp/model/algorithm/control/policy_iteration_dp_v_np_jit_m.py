from __future__ import annotations
# from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange, float64, int64, void


# if TYPE_CHECKING:
#     from mdp.model.environment.environment import Environment
#     from mdp.model.agent.agent import Agent
# from mdp import common
# from mdp.model.algorithm.abstract.dynamic_programming_v import DynamicProgrammingV


@njit(float64[::, ::1](
        int64[::1],
        float64[::, ::, ::1]
        ), parallel=True, cache=True)
def get_state_transition_probability_matrix(policy_vector: np.ndarray,
                                            state_transition_probabilities: np.ndarray
                                            ) -> np.ndarray:
    # policy_vector[s] = a ; π(a|s) deterministic
    # state_transition_probabilities[s, a, s'] = p(s'|s,a)
    # state_transition_probability_matrix[s, s'] = p(s'|s) = Σa π(a|s) . p(s'|s,a)
    # so sum over axis 1 of policy_matrix and axis 1 of self.state_transition_probabilities

    # state_transition_probability_matrix: np.ndarray = np.einsum(
    #     'ij,ijk->ik',
    #     policy_matrix,
    #     state_transition_probabilities
    # )

    n_states: int = state_transition_probabilities.shape[0]
    # n_actions: int = state_transition_probabilities.shape[1]

    out = np.zeros(shape=(n_states, n_states), dtype=np.float64)
    for i in prange(n_states):
        j = policy_vector[i]
        for k in range(n_states):
            out[i, k] += state_transition_probabilities[i, j, k]

    return out


# @njit(float64[::1](
#         int64[::1],
#         float64[::, ::1]
#         ), parallel=True, cache=True)

@njit(parallel=True, cache=True)
def get_reward_vector(policy_vector: np.ndarray,
                      expected_reward: np.ndarray,
                      ) -> np.ndarray:
    # policy_vector[s] = a ; π(a|s) deterministic
    # expected_reward[s,a] = Σs',r p(s',r|s,a).r
    # reward_vector[s] = Σa π(a|s) . Σs',r p(s',r|s,a).r
    # so sum over axis 1 of policy_matrix and axis 1 of expected_reward

    # reward_vector: np.ndarray = np.einsum(
    #     'ij,ij->i',
    #     policy_matrix,
    #     expected_reward
    # )

    n_states = expected_reward.shape[0]
    # n_actions = expected_reward.shape[1]

    reward_vector = np.zeros(shape=n_states, dtype=np.float64)

    for i in prange(n_states):
        j = policy_vector[i]
        reward_vector[i] += expected_reward[i, j]

    return reward_vector


# @njit(void(
#     int64[::1],
#     float64[::, ::1],
#     float64[::1]
#         ), parallel=True, cache=True)
# def fill_reward_vector(policy_vector: np.ndarray,
#                        expected_reward: np.ndarray,
#                        reward_vector: np.ndarray
#                        ):
#     # policy_vector[s] = a ; π(a|s) deterministic
#     # expected_reward[s,a] = Σs',r p(s',r|s,a).r
#     # reward_vector[s] = Σa π(a|s) . Σs',r p(s',r|s,a).r
#     # so sum over axis 1 of policy_matrix and axis 1 of expected_reward
#
#     # reward_vector: np.ndarray = np.einsum(
#     #     'ij,ij->i',
#     #     policy_matrix,
#     #     expected_reward
#     # )
#
#     n_states = expected_reward.shape[0]
#     # n_actions = expected_reward.shape[1]
#
#     # reward_vector = np.zeros(shape=n_states, dtype=np.float64)
#
#     for i in prange(n_states):
#         j = policy_vector[i]
#         reward_vector[i] += expected_reward[i, j]
#
#     # return reward_vector


# @njit(parallel=True, cache=True)

# @njit(float64[::1](
#     float64[::1],
#     float64,
#     float64[::, ::1],
#     float64[::1]
#     ), parallel=True, cache=True)

@njit(parallel=True, cache=True)
def apply_bellman_operator(
        r: np.ndarray,
        gamma: float,
        t: np.ndarray,
        v: np.ndarray
) -> np.ndarray:
    # policy_vector[s] = a ; π(a|s) deterministic
    # expected_reward[s,a] = Σs',r p(s',r|s,a).r
    # reward_vector[s] = Σa π(a|s) . Σs',r p(s',r|s,a).r
    # so sum over axis 1 of policy_matrix and axis 1 of expected_reward

    # reward_vector: np.ndarray = np.einsum(
    #     'ij,ij->i',
    #     policy_matrix,
    #     expected_reward
    # )

    # n_states = expected_reward.shape[0]
    # n_actions = expected_reward.shape[1]
    #
    # reward_vector = np.zeros(shape=n_states, dtype=np.float64)
    #
    # for i in prange(n_states):
    #     j = policy_vector[i]
    #     reward_vector[i] += expected_reward[i, j]

    # new_v = r + gamma * np.dot(t, v)

    n_states = v.shape[0]
    new_v = np.zeros_like(v)
    for i in prange(n_states):
        for j in range(n_states):
            new_v[i] += t[i, j] * v[j]
        new_v[i] *= gamma
        new_v[i] += r[i]

    return new_v

# @njit(types.Tuple((float64[::1], int64, float64))(
#         float64,
#         float64,
#         float64[::1],
#         float64[::, ::1],
#         float64[::, ::, ::1],
#         float64[::, ::1],
#         int64
#         ), parallel=True, cache=True)

# @njit(parallel=False, cache=True)


# @profile

# @njit(cache=True)

# @profile

@njit(cache=True)
def policy_evaluation_algorithm(gamma: float,
                                theta: float,
                                v: np.ndarray,
                                policy_vector: np.ndarray,
                                state_transition_probabilities: np.ndarray,
                                expected_reward: np.ndarray,
                                iteration_timeout: int
                                ):
    iteration: np.int = 1
    cont: np.bool = True
    delta: np.float = 1.0

    # state_transition_probability_matrix
    # T[s, s'] = p(s'|s) = Σa π(a|s).p(s'|s,a)
    # noinspection PyPep8Naming
    T: np.ndarray = get_state_transition_probability_matrix(policy_vector, state_transition_probabilities)
    # r[s] = E[r|s,a=π(a|s)] = Σa π(a|s) Σs',r p(s',r|s,a).r
    # r = np.zeros(shape=policy_vector.shape, dtype=np.float64)
    r: np.ndarray = get_reward_vector(policy_vector, expected_reward)
    # prev_v = np.empty_like(v)

    while cont and delta >= theta and iteration < iteration_timeout:
        prev_v = v.copy()
        # bellman operator v'[s] = Σa π(a|s) Σs',r p(s',r|s,a).(r + γ.v(s'))
        # v = r + γTv
        # tv = np.dot(T, v)
        # gtv = gamma*tv
        # v = r + gtv
        v = r + gamma * np.dot(T, v)
        # v = apply_bellman_operator(r, gamma, T, v)
        # check for convergence
        diff = v - prev_v
        delta = np.linalg.norm(diff, ord=1)

        # if self._verbose:
        #     print(f"iteration = {iteration}\tdelta={delta:.2f}")
        # if do_call_back:
        #     cont = self._step_callback()
        iteration += 1

    return v, iteration, delta


# if __name__ == '__main__':
#     print(test(2.0))
    # test.inspect_types()

    # policy_evaluation_algorithm(gamma=1.0,
    #                             theta=1.0,
    #                             v=np.zeros(shape=2, dtype=np.float64),
    #                             policy_matrix=np.zeros(shape=(2, 3), dtype=np.float64),
    #                             state_transition_probabilities=np.zeros(shape=(2, 3, 2), dtype=np.float64),
    #                             expected_reward=np.zeros(shape=(2, 3), dtype=np.float64),
    #                             iteration_timeout=0)
    # policy_evaluation_algorithm.parallel_diagnostics(level=4)

if __name__ == '__main__':
    # get_state_transition_probability_matrix.parallel_diagnostics(level=4)
    get_reward_vector.parallel_diagnostics(level=4)
    # policy_evaluation_algorithm.parallel_diagnostics(level=4)

    # get_reward_vector.inspect_types()
