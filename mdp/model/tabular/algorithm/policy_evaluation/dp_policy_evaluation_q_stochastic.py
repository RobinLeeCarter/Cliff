from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model.tabular.environment.tabular_environment import TabularEnvironment
    from mdp.model.tabular.agent.agent import Agent
from mdp import common
from mdp.model.tabular.algorithm import linear_algebra as la
from mdp.model.tabular.algorithm.abstract.dynamic_programming_q import DynamicProgrammingQ


class DpPolicyEvaluationQStochastic(DynamicProgrammingQ):
    def __init__(self,
                 environment_: TabularEnvironment,
                 agent: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment_, agent, algorithm_parameters, policy_parameters)
        self._algorithm_type = common.AlgorithmType.DP_POLICY_EVALUATION_Q_STOCHASTIC
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

        # state_transition_p[s, a, s'] = p(s'|s,a)
        state_transition_p: np.ndarray = self._environment.dynamics.state_transition_probabilities

        # expected_reward[s, a] = E[r|s,a] = Σs',r p(s',r|s,a).r
        expected_reward: np.ndarray = self._environment.dynamics.expected_reward

        # Q[s, a]
        q: np.ndarray = self.Q.matrix

        gamma = self._agent.gamma

        while cont and above_theta and iteration < self._iteration_timeout:
            # # v[s'] = Σa' π(a'|s') . q(s', a')
            v: np.ndarray = la.derive_v_from_q(policy_matrix, q)

            # bellman_update_q_deterministic
            # q(s,a) = Σs',r p(s',r|s,a).r  + γ.Σs' p(s'|s,a) Σa' π(a'|s').q(s',a')
            # q(s,a) = Σs',r p(s',r|s,a).r  + γ.Σs' p(s'|s,a) v(s')
            new_q = expected_reward + gamma * np.dot(state_transition_p, v)

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

        if iteration == self._iteration_timeout:
            print(f"Warning: Timed out at {iteration} iterations")
        else:
            if self._verbose:
                print(f"Policy Evaluation completed.")

        self.Q.set_matrix(q)

    # def _derive_v(self,
    #               policy_matrix: np.ndarray,
    #               q: np.ndarray,
    #               s_a_compatibility: np.ndarray
    #               ) -> np.ndarray:
    #     # policy_matrix[s, a] = π(a|s)
    #     # q[s, a] = q(s,a)
    #     # v[s'] = Σa' π(a'|s') . q(s', a')
    #     # so sum over axis 1 of policy_matrix and axis 1 of q
    #
    #     v: np.ndarray = np.einsum(
    #         'ij,ij->i',
    #         policy_matrix,
    #         q
    #     )
    #     return v

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
