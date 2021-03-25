from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model import environment, agent
from mdp import common
from mdp.model.algorithm import abstract, value_function


class MCPrediction(abstract.EpisodicMonteCarlo):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment_, agent_, algorithm_parameters, policy_parameters)
        self._algorithm_type = common.AlgorithmType.MC_PREDICTION
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name} first_visit={self.first_visit}"
        self._create_v()
        self._N = value_function.StateVariable(self._environment, initial_value=0.0)

    def _process_time_step(self, t: int):
        # only do updates on the time-steps that should be done
        if self.first_visit:
            if self._episode.is_first_visit[t]:
                self._update_v(t)
        else:
            self._update_v(t)

    def _update_v(self, t: int):
        state = self._episode[t].state
        target = self._episode.G[t]
        delta = target - self.V[state]
        self._N[state] += 1
        # V[St] = V[St] + (1/N(s)).(G(t) - V(St))
        self.V[state] += delta / self._N[state]
