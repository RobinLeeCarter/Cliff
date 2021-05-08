from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment
    from mdp.model.agent.agent import Agent
    from mdp.model.policy.deterministic import Deterministic
from mdp import common
from mdp.model.algorithm.abstract.dynamic_programming_v import DynamicProgrammingV


class PolicyEvaluationDpVNp(DynamicProgrammingV):
    def __init__(self,
                 environment_: Environment,
                 agent_: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment_, agent_, algorithm_parameters, policy_parameters)
        self._algorithm_type = common.AlgorithmType.POLICY_EVALUATION_DP_V
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

    # @profile
    def _policy_evaluation(self, do_call_back: bool = False):
        iteration: int = 1
        cont: bool = True
        # delta: float = float('inf')
        above_theta: bool = True

        if self._verbose:
            print(f"Starting Policy Evaluation PolicyEvaluationDpVNp ...")

        # policy_matrix[s, a] = π(a|s)
        # noinspection PyTypeChecker
        policy: Deterministic = self._agent.policy
        policy_vector: np.ndarray = policy.get_policy_vector()

        # state_transition_probabilities[s, a, s'] = p(s'|s,a)
        state_transition_probabilities: np.ndarray = self._environment.dynamics.state_transition_probabilities
        # expected_reward_np[s,a] = E[r|s,a] = Σs',r p(s',r|s,a).r
        expected_reward: np.ndarray = self._environment.dynamics.expected_reward_np
        # V[s]
        v: np.ndarray = self.V.vector
        gamma = self._agent.gamma

        # identity
        i = np.arange(policy_vector.shape[0])

        # state_transition_probability_matrix
        # T[s, s'] = p(s'|s) = Σa π(a|s).p(s'|s,a)
        # noinspection PyPep8Naming
        T: np.ndarray = state_transition_probabilities[i, policy_vector, :]
        # T: np.ndarray = self._get_state_transition_probability_matrix(policy_vector, state_transition_probabilities)
        # r[s] = E[r|s,a=π(a|s)] = Σa π(a|s) Σs',r p(s',r|s,a).r
        r: np.ndarray = expected_reward[i, policy_vector]
        # r: np.ndarray = self._get_reward_vector(policy_vector, expected_reward)

        while cont and above_theta and iteration < self._iteration_timeout:
            # bellman operator v'[s] = Σa π(a|s) Σs',r p(s',r|s,a).(r + γ.v(s'))
            # v = r + γTv
            new_v = r + gamma*np.dot(T, v)
            # check for convergence
            # diff = abs(v - prev_v)
            # delta = np.linalg.norm(diff, ord=1)
            above_theta = l1_norm_above(new_v, v, self._theta)
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

    # noinspection PyPep8Naming
    def perform_update(self, v: np.ndarray, r: np.ndarray, T: np.ndarray, gamma: float) -> np.ndarray:
        new_v = r + gamma * np.dot(T, v)
        return new_v


@njit(cache=True)
def l1_norm_above(v1: np.ndarray, v2: np.ndarray, theta: float) -> bool:
    l1_norm: float = 0.0
    above_theta: bool = False
    for i in range(len(v1)):
        l1_norm += abs(v1[i] - v2[i])
        if l1_norm > theta:
            above_theta: bool = True
            break
    return above_theta


# if __name__ == '__main__':
    # partially_inplace_update.inspect_types()
    #     partially_inplace_update.parallel_diagnostics(level=4)
