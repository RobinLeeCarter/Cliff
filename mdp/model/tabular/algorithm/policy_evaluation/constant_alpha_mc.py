from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.tabular.agent.tabular_agent import TabularAgent
from mdp import common
from mdp.model.tabular.algorithm.abstract.episodic_monte_carlo import EpisodicMonteCarlo


class ConstantAlphaMC(EpisodicMonteCarlo):
    def __init__(self,
                 agent: TabularAgent,
                 algorithm_parameters: common.AlgorithmParameters,
                 name: str
                 ):
        super().__init__(agent, algorithm_parameters, name)
        self._alpha = self._algorithm_parameters.alpha
        self.title = f"{self.name} α={self._alpha}"
        self._create_v()

    def _process_time_step(self, t: int):
        s = self._episode[t].s
        target = self._episode.G[t]
        delta = target - self.V[s]
        self.V[s] += self._alpha * delta
