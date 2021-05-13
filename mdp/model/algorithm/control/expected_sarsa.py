from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment
    from mdp.model.agent.agent import Agent
from mdp import common
from mdp.model.algorithm.abstract.episodic_online_control import EpisodicOnlineControl


class ExpectedSarsa(EpisodicOnlineControl):
    def __init__(self,
                 environment_: Environment,
                 agent_: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment_, agent_, algorithm_parameters, policy_parameters)
        self._alpha = self._algorithm_parameters.alpha
        self._algorithm_type = common.AlgorithmType.EXPECTED_SARSA
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name} α={self._alpha}"
        self._create_q()

    def _do_training_step(self):
        ag = self._agent
        ag.choose_action()
        ag.take_action()

        q_expectation_over_a = self._get_expectation_over_a(ag.s)
        target = ag.r + self._gamma * q_expectation_over_a
        delta = target - self.Q[ag.prev_s, ag.prev_a]

        # print(f"s: {ag.prev_s} \ta: {ag.prev_a} \ts' {ag.s}")
        # print(f"E: {q_expectation_over_a}")
        # print(f"t: {target}")
        # print(f"d: {delta}")

        # print(f"Before:")
        # print(f"argmax:   {self.Q.argmax[ag.prev_s]}")
        # print(f"max:      {self.Q.max[ag.prev_s]}")
        # print(f"Q:        {self.Q[ag.prev_s, ag.prev_a]}")

        self.Q[ag.prev_s, ag.prev_a] += self._alpha * delta
        # print(f"After:")
        # print(f"argmax:   {self.Q.argmax[ag.prev_s]}")
        # print(f"max:      {self.Q.max[ag.prev_s]}")
        # print(f"Q:        {self.Q[ag.prev_s, ag.prev_a]}")
        # update policy to be in-line with Q
        self._agent.policy[ag.prev_s] = self.Q.argmax[ag.prev_s]
        # print()

    def _get_expectation_over_a(self, s: int) -> float:
        probability_vector: np.ndarray = self._agent.policy.get_probability_vector(s)
        q_slice: np.ndarray = self.Q.matrix[s, :]
        expectation: float = float(np.dot(probability_vector, q_slice))
        return expectation

        # expectation: float = 0.0
        # for action in self._environment.actions:
        #     probability = self._agent.policy.get_probability(state, action)
        #     expectation += probability * self.Q[state, action]
        # return expectation
