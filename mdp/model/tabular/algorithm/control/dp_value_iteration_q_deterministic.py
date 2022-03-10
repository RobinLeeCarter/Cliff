from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model.tabular.environment.tabular_environment import TabularEnvironment
    from mdp.model.tabular.agent.agent import Agent
from mdp import common
from mdp.model.tabular.algorithm import linear_algebra as la
from mdp.model.tabular.algorithm.abstract.dynamic_programming_q import DynamicProgrammingQ


class DpValueIterationQ(DynamicProgrammingQ):
    def __init__(self,
                 environment: TabularEnvironment,
                 agent: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 name: str
                 ):
        super().__init__(environment, agent, algorithm_parameters, name)
        self.title = f"{self.name} θ={self._theta}"

    def run(self):
        do_call_back: bool = bool(self._step_callback)
        self._value_iteration(do_call_back)

    def _value_iteration(self, do_call_back: bool = False):
        # policy_: policy.Policy = self._agent.target_policy
        # assert isinstance(policy_, policy.Deterministic)
        iteration: int = 1
        cont: bool = True
        # delta: float = float('inf')
        above_theta: bool = True

        if self._verbose:
            print(f"Starting Value Iteration ...")

        # state_transition_p[s, a, s'] = p(s'|s,a)
        state_transition_p: np.ndarray = self._environment.dynamics.state_transition_probabilities

        # expected_reward_np[s,a] = E[r|s,a] = Σs',r p(s',r|s,a).r
        expected_reward: np.ndarray = self._environment.dynamics.expected_reward

        # Q[s, a]
        q: np.ndarray = self.Q.matrix

        gamma = self._agent.gamma

        while cont and above_theta and iteration < self._iteration_timeout:
            # value iteration: q[s] <- Σs',r p(s',r|s,a).( r + γ.Max_over_a'[(q(s',a')] )
            #                        = Σs',r p(s',r|s,a).r + γ.Σs' p(s'|s,a).Max_over_a'[q(s',a')]

            # q_max[s] = Max_over_a'[q(s',a')]
            q_max: np.ndarray = np.max(q, axis=1)

            # expected_reward[s, a] = Σs',r p(s',r|s,a).r
            # state_transition_p[s, a, s'] = p(s'|s,a)
            # new_q = Σs',r p(s',r|s,a).r + γ.Σs' p(s'|s,a).Max_over_a'[q(s',a')]
            new_q: np.ndarray = expected_reward + gamma * np.dot(state_transition_p, q_max)

            # check for convergence
            # diff = abs(v - prev_v)
            # delta = np.linalg.norm(diff, ord=1)
            above_theta = la.l1_norm_above(new_q, q, self._theta)
            q = new_q

            if self._verbose:
                print(f"iteration = {iteration}")
                # print(f"iteration = {iteration}\tdelta={delta:.2f}")
            if do_call_back:
                cont = self._step_callback()
            iteration += 1

        theta_decimal_places: int = -math.ceil(math.log10(self._theta))
        q = q.round(theta_decimal_places)

        self.Q.set_matrix(q)

        # π'(s) = argmax_over_a( q(s, a) )
        new_policy_vector: np.ndarray = self.Q.argmax.copy()

        self._agent.policy.set_policy_vector(new_policy_vector)

        # policy_vector = np.argmax(q, axis=1)
        #
        # # policy_vector = make_policy_greedy_wrt_v(v, expected_reward, state_transition_p, gamma, round_first=True)
        # self._agent.policy.set_policy_vector(policy_vector)

        if iteration == self._iteration_timeout:
            print(f"Warning: Timed out at {iteration} iterations")
        else:
            if self._verbose:
                print(f"Value Iteration completed ...")
                # print(f"Value Iteration completed. delta={delta:.2f}")
        # print(f"iterations: {iteration - 1}")


# @njit(cache=True)
# def perform_update(v: np.ndarray, expected_reward: np.ndarray, state_transition_p: np.ndarray, gamma: float)\
#         -> np.ndarray:
#     # expected_reward[s, a] = Σs',r p(s',r|s,a).r
#     # state_transition_p[s, a, s'] = p(s'|s,a)
#
#     # value iteration: v'[s] = Max_over_a Σs',r p(s',r|s,a).(r + γ.v(s'))
#     #                        = Max_over_a [ Σs',r p(s',r|s,a).r  +  γ Σs' p(s'|s,a).v(s') ]
#     # expected_return[s,a] =                Σs',r p(s',r|s,a).r  +  γ Σs' p(s'|s,a).v(s')
#     expected_return: np.ndarray = expected_reward + gamma * np.dot(state_transition_p, v)
#     return np.max(expected_return, axis=1)


# @njit(cache=True)
# def make_policy_greedy_wrt_v(v: np.ndarray, expected_reward: np.ndarray, state_transition_p: np.ndarray
# , gamma: float)\
#         -> np.ndarray:
    # expected_reward[s, a] = Σs',r p(s',r|s,a).r
    # state_transition_p[s, a, s'] = p(s'|s,a)

    # value iteration: v'[s] = Max_over_a Σs',r p(s',r|s,a).(r + γ.v(s'))
    #                        = Max_over_a [ Σs',r p(s',r|s,a).r  +  γ Σs' p(s'|s,a).v(s') ]
    # expected_return[s,a] =                Σs',r p(s',r|s,a).r  +  γ Σs' p(s'|s,a).v(s')
    # expected_return: np.ndarray = expected_reward + gamma * np.dot(state_transition_p, v)
    # return expected_return.argmax(axis=1)


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
