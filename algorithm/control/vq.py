from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import environment
    import agent
import common
from algorithm import abstract


class VQ(abstract.EpisodicOnline):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 ):
        super().__init__(environment_, agent_, algorithm_parameters)
        self._alpha_variable: bool = self._algorithm_parameters.alpha_variable
        self._alpha: float = self._algorithm_parameters.alpha
        self._algorithm_type = common.AlgorithmType.VQ
        self.name = common.algorithm_name[self._algorithm_type]

        if self._alpha_variable:
            self.title = f"{self.name} α=0.5 then α=0.1"
        else:
            self.title = f"{self.name} α={self._alpha}"

    def parameter_changes(self, iteration: int):
        if self._alpha_variable:
            if iteration <= 50:
                self._alpha = 0.5
            else:
                self._alpha = 0.1

            # if iteration <= 20:
            #     self._alpha = 0.5
            # else:
            #     self._alpha = 10/iteration

    def _do_training_step(self):
        self._agent.choose_action()
        self._agent.take_action()

        prev_state = self._agent.prev_state
        prev_action = self._agent.prev_action
        reward = self._agent.reward
        state = self._agent.state

        target = reward + self._gamma * self._V[state]

        v_delta = target - self._V[prev_state]
        self._V[prev_state] += self._alpha * v_delta

        q_delta = target - self._Q[prev_state, prev_action]
        self._Q[prev_state, prev_action] += self._alpha * q_delta

        # update policy to be in-line with Q
        self._agent.policy[prev_state] = self._Q.argmax_over_actions(prev_state)
