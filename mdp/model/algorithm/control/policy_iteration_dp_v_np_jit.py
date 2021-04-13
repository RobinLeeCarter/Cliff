from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment
    from mdp.model.agent.agent import Agent
    from mdp.model.policy.deterministic import Deterministic
from mdp import common
from mdp.model.algorithm.abstract.dynamic_programming_v import DynamicProgrammingV
from mdp.model.algorithm.control import policy_iteration_dp_v_np_jit_v as jit


class PolicyIterationDpVNpJit(DynamicProgrammingV):
    def __init__(self,
                 environment_: Environment,
                 agent_: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment_, agent_, algorithm_parameters, policy_parameters)
        self._algorithm_type = common.AlgorithmType.POLICY_ITERATION_DP_V
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name} θ={self._theta}"

    def run(self):
        # policy_: policy.Policy = self._agent.target_policy
        # assert isinstance(policy_, policy.Deterministic)

        if self._verbose:
            print(f"Starting Policy Iteration ...")

        iteration: int = 1
        policy_stable: bool = False
        cont: bool = True
        if self._step_callback:
            cont = self._step_callback()
        while cont and not policy_stable and iteration < self._iteration_timeout:
            if self._verbose:
                print(f"Policy Iteration. Iteration = {iteration}")
            self._policy_evaluation()
            policy_stable = self._policy_improvement()
            if self._step_callback:
                cont = self._step_callback()
            iteration += 1

        if iteration == self._iteration_timeout:
            print(f"Warning: Timed out at {iteration} iterations")
        else:
            if self._verbose:
                print(f"Policy Iteration completed ...")

    def _policy_evaluation(self, do_call_back: bool = False):
        if self._verbose:
            print(f"Starting Policy Evaluation PolicyEvaluationDpVNpJit ...")

        # policy_matrix[s, a] = π(a|s)
        # policy_matrix: np.ndarray = self._agent.policy.get_policy_matrix()
        # policy_vector[s] = a ; π(a|s) deterministic
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

        v, iteration, delta = jit.policy_evaluation_algorithm(
            gamma,
            self._theta,
            v,
            policy_vector,
            state_transition_probabilities,
            expected_reward,
            self._iteration_timeout)

        # jit.policy_evaluation_algorithm.parallel_diagnostics(level=4)
        # above line works ;-) but the below works
        # x = jit.policy_evaluation_algorithm
        # v, iteration, delta = x(
        #     gamma,
        #     self._theta,
        #     v,
        #     policy_matrix,
        #     state_transition_probabilities,
        #     expected_reward,
        #     self._iteration_timeout)
        # x.parallel_diagnostics(level=4)
        # exit()

        if iteration == self._iteration_timeout:
            print(f"Warning: Timed out at {iteration} iterations")
        else:
            if self._verbose:
                print(f"Policy Evaluation completed. delta={delta:.2f}")

        self.V.vector = v

    # def _policy_evaluation_algorithm(self,
    #                                  gamma: float,
    #                                  theta: float,
    #                                  v: np.ndarray,
    #                                  policy_matrix: np.ndarray,
    #                                  state_transition_probabilities: np.ndarray,
    #                                  expected_reward: np.ndarray,
    #                                  iteration_timeout: int
    #                                  ):
    #     iteration: int = 1
    #     cont: bool = True
    #     delta: float = float('inf')
    #
    #     # state_transition_probability_matrix
    #     # T[s, s'] = p(s'|s) = Σa π(a|s).p(s'|s,a)
    #     # noinspection PyPep8Naming
    #     T: np.ndarray = jit.get_state_transition_probability_matrix(policy_matrix, state_transition_probabilities)
    #     # r[s] = E[r|s,a=π(a|s)] = Σa π(a|s) Σs',r p(s',r|s,a).r
    #     r: np.ndarray = jit.get_reward_vector(policy_matrix, expected_reward)
    #     # prev_v = np.empty_like(v)
    #
    #     while cont and delta >= theta and iteration < iteration_timeout:
    #         prev_v = v.copy()
    #         # bellman operator v'[s] = Σa π(a|s) Σs',r p(s',r|s,a).(r + γ.v(s'))
    #         # v = r + γTv
    #         v = r + gamma*np.dot(T, v)
    #         # check for convergence
    #         diff = v - prev_v
    #         delta = np.linalg.norm(diff, ord=1)
    #
    #         if self._verbose:
    #             print(f"iteration = {iteration}\tdelta={delta:.2f}")
    #         # if do_call_back:
    #         #     cont = self._step_callback()
    #         iteration += 1
    #
    #     return v, iteration, delta

    # def _get_state_transition_probability_matrix(self,
    #                                              policy_matrix: np.ndarray,
    #                                              state_transition_probabilities: np.ndarray
    #                                              ) -> np.ndarray:
    #     # policy_matrix[s, a] = π(a|s)
    #     # state_transition_probabilities[s, a, s'] = p(s'|s,a)
    #     # state_transition_probability_matrix[s, s'] = p(s'|s) = Σa π(a|s) . p(s'|s,a)
    #     # so sum over axis 1 of policy_matrix and axis 1 of self.state_transition_probabilities
    #     state_transition_probability_matrix: np.ndarray = np.einsum(
    #         'ij,ijk->ik',
    #         policy_matrix,
    #         state_transition_probabilities
    #     )
    #     return state_transition_probability_matrix

    # def _get_reward_vector(self,
    #                        policy_matrix: np.ndarray,
    #                        expected_reward: np.ndarray
    #                        ) -> np.ndarray:
    #     # policy_matrix[s,a] = π(a|s)
    #     # expected_reward[s,a] = Σs',r p(s',r|s,a).r
    #     # reward_vector[s] = Σa π(a|s) . Σs',r p(s',r|s,a).r
    #     # so sum over axis 1 of policy_matrix and axis 1 of expected_reward
    #
    #     reward_vector: np.ndarray = np.einsum(
    #         'ij,ij->i',
    #         policy_matrix,
    #         expected_reward
    #     )
    #     return reward_vector

    def _policy_improvement(self, do_call_back: bool = False) -> bool:
        # assert isinstance(policy_, policy.Deterministic)
        # policy_: policy.Deterministic

        if self._verbose:
            print(f"Starting Policy Improvement ...")

        # # policy_matrix[s, a] = π(a|s)
        # policy_matrix: np.ndarray = self._agent.policy.get_policy_matrix()

        # policy_vector[s] = a ; π(a|s) deterministic
        # noinspection PyTypeChecker
        policy: Deterministic = self._agent.policy
        policy_vector: np.ndarray = policy.get_policy_vector()
        # old_policy_vector: np.ndarray = policy_vector.copy()
        # old_policy_vector: np.ndarray = policy_matrix.argmax(axis=1)
        # state_transition_probabilities[s, a, s'] = p(s'|s,a)
        state_transition_probabilities: np.ndarray = self._environment.dynamics.state_transition_probabilities
        # expected_reward_np[s,a] = E[r|s,a] = Σs',r p(s',r|s,a).r
        expected_reward: np.ndarray = self._environment.dynamics.expected_reward_np
        # V[s]
        v: np.ndarray = self.V.vector
        gamma = self._agent.gamma

        # expected_return[s,a] = ( Σs',r p(s',r|s,a).r ) + ( γ . Σs' p(s'|s,a).v(s') )
        expected_return: np.ndarray = expected_reward + gamma * np.dot(state_transition_probabilities, v)

        # argmax(a) Σs',r p(s',r|s,a).(r + γ.v(s'))
        new_policy_vector: np.ndarray = expected_return.argmax(axis=1)
        policy_stable: bool = np.allclose(policy_vector, new_policy_vector)
        self._agent.policy.set_policy_vector(new_policy_vector)

        if self._verbose:
            print(f"Policy Improvement completed. policy_stable = {policy_stable}")
        if do_call_back:
            self._step_callback()

        return policy_stable

