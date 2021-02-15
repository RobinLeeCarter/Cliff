from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import environment
    import agent
import common
from algorithm import abstract


class QLearning(abstract.EpisodicOnline):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(environment_, agent_, algorithm_parameters)
        self._alpha = self._algorithm_parameters.alpha
        self._algorithm_type = common.AlgorithmType.Q_LEARNING
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name} α={self._alpha}"

    def _do_training_step(self):
        self._agent.choose_action()
        self._agent.take_action()

        prev_state = self._agent.prev_state
        prev_action = self._agent.prev_action
        reward = self._agent.reward
        state = self._agent.state

        q_max_over_a = self._Q.max_over_actions(state)
        target = reward + self._gamma * q_max_over_a
        delta = target - self._Q[prev_state, prev_action]
        self._Q[prev_state, prev_action] += self._alpha * delta
        # update policy to be in-line with Q
        self._agent.policy[prev_state] = self._Q.argmax_over_actions(prev_state)
