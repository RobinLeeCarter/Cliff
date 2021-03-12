from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model import environment, agent
from mdp import common
from mdp.model.algorithm import abstract


class Sarsa(abstract.EpisodicOnline):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment_, agent_, algorithm_parameters, policy_parameters)
        self._alpha = self._algorithm_parameters.alpha
        self._algorithm_type = common.AlgorithmType.SARSA
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name} α={self._alpha}"
        self._create_q()

    def initialize(self):
        super().initialize()
        self._make_policy_greedy_wrt_q()

    def _start_episode(self):
        self._agent.choose_action()

    def _do_training_step(self):
        self._agent.take_action()
        self._agent.choose_action()

        prev_state: environment.State = self._agent.prev_state
        prev_action: environment.Action = self._agent.prev_action
        reward: float = self._agent.reward
        state: environment.State = self._agent.state
        action: environment.Action = self._agent.action

        target = reward + self._gamma * self.Q[state, action]
        delta = target - self.Q[prev_state, prev_action]
        self.Q[prev_state, prev_action] += self._alpha * delta
        # update policy to be in-line with Q
        self._agent.policy[prev_state] = self.Q.argmax_over_actions(prev_state)
