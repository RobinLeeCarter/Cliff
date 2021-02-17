from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model import environment, agent
from mdp import common
from mdp.model.algorithm import abstract


class TD0(abstract.EpisodicOnline):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(environment_, agent_, algorithm_parameters)
        self._alpha = self._algorithm_parameters.alpha
        self._algorithm_type = common.AlgorithmType.CONSTANT_ALPHA_MC
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name} α={self._alpha}"

    def _do_training_step(self):
        self._agent.choose_action()
        self._agent.take_action()

        prev_state = self._agent.prev_state
        reward = self._agent.reward
        state = self._agent.state

        target = reward + self._gamma * self._V[state]
        delta = target - self._V[prev_state]
        self._V[prev_state] += self._alpha * delta
