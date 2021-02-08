from __future__ import annotations
from typing import TYPE_CHECKING

import constants
from algorithm import abstract
if TYPE_CHECKING:
    import environment
    import agent


class Sarsa(abstract.EpisodicOnline):
    name: str = "Sarsa"

    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 alpha: float = 0.5,
                 verbose: bool = False
                 ):
        super().__init__(environment_, agent_, verbose)
        self.title = f"{Sarsa.name} α={alpha}"
        self._alpha = alpha

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

        target = reward + constants.GAMMA * self._Q[state, action]
        delta = target - self._Q[prev_state, prev_action]
        self._Q[prev_state, prev_action] += self._alpha * delta
        # update policy to be in-line with Q
        self.agent.policy[prev_state] = self._Q.argmax_over_actions(prev_state)
