from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import environment
    import agent
import common
from algorithm import abstract


class VQ(abstract.EpisodicOnline):
    name: str = "VQ"

    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 alpha: float = 0.5,
                 alpha_variable: bool = False,
                 verbose: bool = False
                 ):
        self.alpha_variable = alpha_variable
        super().__init__(environment_, agent_, verbose)
        if self.alpha_variable:
            self.title = f"{VQ.name} α=0.5 then α=0.1"
        else:
            self.title = f"{VQ.name} α={alpha}"
        self._alpha = alpha

    def parameter_changes(self, iteration: int):
        if self.alpha_variable:
            if iteration <= 50:
                self._alpha = 0.5
            else:
                self._alpha = 0.1

            # if iteration <= 20:
            #     self._alpha = 0.5
            # else:
            #     self._alpha = 10/iteration

    def _do_training_step(self):
        self.agent.choose_action()
        self.agent.take_action()

        prev_state = self.agent.prev_state
        prev_action = self.agent.prev_action
        reward = self.agent.reward
        state = self.agent.state

        target = reward + common.GAMMA * self._V[state]

        v_delta = target - self._V[prev_state]
        self._V[prev_state] += self._alpha * v_delta

        q_delta = target - self._Q[prev_state, prev_action]
        self._Q[prev_state, prev_action] += self._alpha * q_delta

        # update policy to be in-line with Q
        self.agent.policy[prev_state] = self._Q.argmax_over_actions(prev_state)
