from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model import environment, agent
from mdp import common
from mdp.model.algorithm import abstract


class OffPolicyMcControl(abstract.EpisodicMonteCarlo):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(environment_, agent_, algorithm_parameters)
        self._algorithm_type = common.AlgorithmType.OFF_POLICY_MC_CONTROL
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name}"
        self._create_q()

    def _process_time_step(self, t: int):
        state = self._episode[t].state
        target = self._episode.G[t]
        delta = target - self.V[state]
        self.V[state] += self._alpha * delta
