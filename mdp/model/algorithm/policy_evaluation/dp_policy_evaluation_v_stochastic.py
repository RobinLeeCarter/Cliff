from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model.environment.environment_tabular import EnvironmentTabular
    from mdp.model.agent.agent import Agent
from mdp import common
from mdp.model.algorithm import linear_algebra as la
from mdp.model.algorithm.abstract.dynamic_programming_v import DynamicProgrammingV


class DpPolicyEvaluationVStochastic(DynamicProgrammingV):
    def __init__(self,
                 environment_: EnvironmentTabular,
                 agent: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment_, agent, algorithm_parameters, policy_parameters)
        self._algorithm_type = common.AlgorithmType.DP_POLICY_EVALUATION_V_STOCHASTIC
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name} θ={self._theta}"

        # # policy_matrix[s, a] = π(a|s)
        # self.policy_matrix: np.ndarray = np.array([], float)
        # # self.state_transition_probabilities[s, a, s'] = p(s'|s,a)
        # self.state_transition_probabilities: np.ndarray = np.array([], float)

    def run(self):
        do_call_back: bool = bool(self._step_callback)
        self._policy_evaluation(do_call_back)
        # if self._verbose:
        #     self.V.print_all_values()

    def _policy_evaluation(self, do_call_back: bool = False):
        iteration: int = 1
        cont: bool = True
        # delta: float = float('inf')
        above_theta: bool = True

        if self._verbose:
            print(f"Starting Policy Evaluation PolicyEvaluationDpVNp ...")

        # policy_matrix[s, a] = π(a|s)
        policy_matrix: np.ndarray = self._agent.policy.get_probability_matrix()

        # state_transition_probabilities[s, a, s'] = p(s'|s,a)
        state_transition_probabilities: np.ndarray = self._environment.dynamics.state_transition_probabilities

        # expected_reward[s, a] = E[r|s,a] = Σs',r p(s',r|s,a).r
        expected_reward: np.ndarray = self._environment.dynamics.expected_reward

        # V[s]
        v: np.ndarray = self.V.vector
        gamma = self._agent.gamma

        # state_transition_probability_matrix
        # T[s, s'] = p(s'|s) = Σa π(a|s).p(s'|s,a)
        # noinspection PyPep8Naming
        T: np.ndarray = self._get_state_transition_probability_matrix(policy_matrix, state_transition_probabilities)
        # r[s] = E[r|s,a=π(a|s)] = Σa π(a|s) Σs',r p(s',r|s,a).r
        r: np.ndarray = self._get_reward_vector(policy_matrix, expected_reward)

        while cont and above_theta and iteration < self._iteration_timeout:
            # prev_v = v.copy()
            # bellman operator v'[s] = Σa π(a|s) Σs',r p(s',r|s,a).(r + γ.v(s'))
            # v = r + γTv
            new_v = la.bellman_update_v(v, r, T, gamma)
            # v = r + gamma*np.dot(T, v)
            # check for convergence
            # diff = v - prev_v
            above_theta = la.l1_norm_above(new_v, v, self._theta)
            # delta = np.linalg.norm(diff, ord=1)
            v = new_v

            if self._verbose:
                print(f"iteration = {iteration}")
            if do_call_back:
                cont = self._step_callback()
            iteration += 1

        if iteration == self._iteration_timeout:
            print(f"Warning: Timed out at {iteration} iterations")
        else:
            if self._verbose:
                print(f"Policy Evaluation completed.")

        self.V.vector = v

    def _get_state_transition_probability_matrix(self,
                                                 policy_matrix: np.ndarray,
                                                 state_transition_probabilities: np.ndarray
                                                 ) -> np.ndarray:
        # policy_matrix[s, a] = π(a|s)
        # state_transition_probabilities[s, a, s'] = p(s'|s,a)
        # state_transition_probability_matrix[s, s'] = p(s'|s) = Σa π(a|s) . p(s'|s,a)
        # so sum over axis 1 of policy_matrix and axis 1 of self.state_transition_probabilities
        state_transition_probability_matrix: np.ndarray = np.einsum(
            'ij,ijk->ik',
            policy_matrix,
            state_transition_probabilities
        )
        return state_transition_probability_matrix

    def _get_reward_vector(self,
                           policy_matrix: np.ndarray,
                           expected_reward: np.ndarray
                           ) -> np.ndarray:
        # policy_matrix[s, a] = π(a|s)
        # expected_reward[s, a] = Σs',r p(s',r|s,a).r
        # reward_vector[s] = Σa π(a|s) . Σs',r p(s',r|s,a).r
        # so sum over axis 1 of policy_matrix and axis 1 of expected_reward

        reward_vector: np.ndarray = np.einsum(
            'ij,ij->i',
            policy_matrix,
            expected_reward
        )
        return reward_vector

    # def _get_reward_vector_looping(self,
    #                                policy_matrix: np.ndarray,
    #                                expected_reward: np.ndarray
    #                                ) -> np.ndarray:
    #     # policy_matrix[s,a] = π(a|s)
    #     # expected_reward[s,a] = Σs',r p(s',r|s,a).r
    #     # reward_vector[s] = Σa π(a|s) . Σs',r p(s',r|s,a).r
    #     # so sum over axis 1 of policy_matrix and axis 1 of expected_reward
    #
    #     states = policy_matrix.shape[0]
    #     actions = policy_matrix.shape[1]
    #
    #     reward_vector: np.ndarray = np.zeros(shape=policy_matrix.shape[0])
    #
    #     for s in range(states):
    #         for a in range(actions):
    #             reward_vector[s] += policy_matrix[s, a] * expected_reward[s, a]
    #
    #     return reward_vector
