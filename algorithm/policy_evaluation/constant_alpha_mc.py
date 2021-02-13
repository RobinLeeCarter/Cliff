from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import environment
    import agent
import common
from algorithm import abstract


class ConstantAlphaMC(abstract.EpisodicMonteCarlo):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 algorithm_parameters: dict[str, any]
                 ):
        super().__init__(environment_, agent_, algorithm_parameters)
        self._alpha = algorithm_parameters['alpha']
        self.algorithm_type = common.AlgorithmType.CONSTANT_ALPHA_MC
        self.name = common.algorithm_name[self.algorithm_type]
        self.title = f"{self.name} α={self._alpha}"

    def _process_time_step(self, t: int):
        state = self.episode[t].state
        target = self.episode.G[t]
        delta = target - self._V[state]
        self._V[state] += self._alpha * delta
