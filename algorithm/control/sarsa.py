from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import environment
    import agent
import common
from algorithm import abstract


class Sarsa(abstract.EpisodicOnline):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 algorithm_parameters: dict[str, any]
                 ):
        super().__init__(environment_, agent_, algorithm_parameters)
        self._alpha = algorithm_parameters['alpha']
        self.algorithm_type = common.AlgorithmType.SARSA
        self.name = common.algorithm_name[self.algorithm_type]
        self.title = f"{self.name} α={self._alpha}"

    def _start_episode(self):
        self.agent.choose_action()

    def _do_training_step(self):
        self.agent.take_action()
        self.agent.choose_action()

        prev_state = self.agent.prev_state
        prev_action = self.agent.prev_action
        reward = self.agent.reward
        state = self.agent.state
        action = self.agent.action

        target = reward + self.gamma * self._Q[state, action]
        delta = target - self._Q[prev_state, prev_action]
        self._Q[prev_state, prev_action] += self._alpha * delta
        # update policy to be in-line with Q
        self.agent.policy[prev_state] = self._Q.argmax_over_actions(prev_state)
