from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model.tabular.agent.tabular_agent import TabularAgent
from mdp import common
from mdp.model.tabular.algorithm import linear_algebra as la
from mdp.model.tabular.algorithm.abstract.dynamic_programming_v import DynamicProgrammingV


class DpPolicyEvaluationVDeterministic(DynamicProgrammingV):
    def __init__(self,
                 agent: TabularAgent,
                 algorithm_parameters: common.AlgorithmParameters,
                 name: str
                 ):
        super().__init__(agent, algorithm_parameters, name)
        # # policy_matrix[s, a] = π(a|s)
        # self.policy_matrix: np.ndarray = np.array([], float)
        # # self.state_transition_probabilities[s, a, s'] = p(s'|s,a)
        # self.state_transition_probabilities: np.ndarray = np.array([], float)

    def run(self):
        do_call_back: bool = bool(self._step_callback)
        self._policy_evaluation(do_call_back)
        # if self._verbose:
        #     self.V.print_all_values()

    # @profile
    def _policy_evaluation(self, do_call_back: bool = False):
        iteration: int = 1
        cont: bool = True
        # delta: float = float('inf')
        above_theta: bool = True

        if self._verbose:
            print(f"Starting Policy Evaluation PolicyEvaluationDpVNp ...")

        # policy_vector[s] = π(s)
        policy_vector: np.ndarray = self._target_policy.get_policy_vector()

        # state_transition_p[s, a, s'] = p(s'|s,a)
        state_transition_p: np.ndarray = self._environment.dynamics.state_transition_probabilities

        # expected_reward[s,a] = E[r|s,a] = Σs',r p(s',r|s,a).r
        expected_reward: np.ndarray = self._environment.dynamics.expected_reward

        # V[s]
        v: np.ndarray = self.V.vector
        gamma = self._agent.gamma

        # identity
        i = np.arange(policy_vector.shape[0])

        # state_transition_probability_matrix
        # T[s, s'] = p(s'|s) = Σa π(a|s).p(s'|s,a)
        # noinspection PyPep8Naming
        T: np.ndarray = state_transition_p[i, policy_vector, :]
        # T: np.ndarray = self._get_state_transition_probability_matrix(policy_vector, state_transition_p)
        # r[s] = E[r|s,a=π(a|s)] = Σa π(a|s) Σs',r p(s',r|s,a).r
        r: np.ndarray = expected_reward[i, policy_vector]
        # r: np.ndarray = self._get_reward_vector(policy_vector, expected_reward)

        # v = converge_v(v, r, T, gamma, self._theta, self._iteration_timeout)

        while cont and above_theta and iteration < self._iteration_timeout:
            # bellman operator v'[s] = Σa π(a|s) Σs',r p(s',r|s,a).(r + γ.v(s'))
            # v = r + γTv
            new_v = la.bellman_update_v(v, r, T, gamma)
            # new_v = r + gamma*np.dot(T, v)
            # check for convergence
            # diff = abs(v - prev_v)
            # delta = np.linalg.norm(diff, ord=1)
            above_theta = la.l1_norm_above(new_v, v, self._theta)
            v = new_v

            if self._verbose:
                print(f"iteration = {iteration}")
                # print(f"iteration = {iteration}\tdelta={delta:.2f}")
            if do_call_back:
                cont = self._step_callback()
            iteration += 1

        if iteration == self._iteration_timeout:
            print(f"Warning: Timed out at {iteration} iterations")
        else:
            if self._verbose:
                print(f"Policy Evaluation completed.")
                # print(f"Policy Evaluation completed. delta={delta:.2f}")
        # print(f"iterations: {iteration - 1}")

        self.V.vector = v
        # print("V assigned")


# noinspection PyPep8Naming
# @njit(cache=True)
# def perform_update(v: np.ndarray, r: np.ndarray, T: np.ndarray, gamma: float) -> np.ndarray:
#     return r + gamma * np.dot(T, v)
#
#
# @njit(cache=True)
# def l1_norm_above(v1: np.ndarray, v2: np.ndarray, theta: float) -> bool:
#     l1_norm: float = 0.0
#     above_theta: bool = False
#     for i in range(len(v1)):
#         l1_norm += abs(v1[i] - v2[i])
#         if l1_norm > theta:
#             above_theta: bool = True
#             break
#     return above_theta


# UNUSED marginally faster for larger runs (10,000 times)
# noinspection PyPep8Naming
# @njit(cache=True)
# def converge_v(v: np.ndarray, r: np.ndarray, T: np.ndarray, gamma: float,
#                theta: float, iteration_timeout: int) -> np.ndarray:
#     above_theta: bool = True
#     iteration: int = 0
#     v1 = v
#     v2 = v.copy()
#     is_v1_update: bool = False
#     while above_theta and iteration < iteration_timeout:
#         iteration += 1
#         is_v1_update = not is_v1_update
#         if is_v1_update:
#             # slower
#             # np.dot(T, v2, out=v1)
#             # v1 = r + gamma * v1
#
#             # fastest
#             v1 = r + gamma * np.dot(T, v2)
#
#             # slower
#             # v1 = perform_update(v2, r, T, gamma)
#         else:
#             # np.dot(T, v1, out=v2)
#             # v2 = r + gamma * v2
#             v2 = r + gamma * np.dot(T, v1)
#             # v2 = perform_update(v1, r, T, gamma)
#         above_theta = l1_norm_above(v1, v2, theta)
#     if iteration == iteration_timeout:
#         print("Warning: Timed out at max iterations")
#     if is_v1_update:
#         return v1
#     else:
#         return v2

# if __name__ == '__main__':
    # partially_inplace_update.inspect_types()
    #     partially_inplace_update.parallel_diagnostics(level=4)
