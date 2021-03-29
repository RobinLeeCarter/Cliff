from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment
    from mdp.model.agent.agent import Agent
from mdp import common
from mdp.model.algorithm import abstract, value_function


class MCPredictionV(abstract.EpisodicMonteCarlo):
    def __init__(self,
                 environment_: Environment,
                 agent_: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment_, agent_, algorithm_parameters, policy_parameters)
        self._algorithm_type = common.AlgorithmType.MC_PREDICTION_V
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name} first_visit={self.first_visit}"
        self._create_v()
        self._N = value_function.StateVariable(self._environment, initial_value=0.0)

    def initialize(self):
        super().initialize()
        self._environment.initialize_policy(self._agent.policy, self._policy_parameters)

    def _process_time_step(self, t: int):
        # only do updates on the time-steps that should be done
        if (self.first_visit and self._episode.is_first_visit[t]) \
                or not self.first_visit:
            state = self._episode[t].state
            target = self._episode.G[t]
            delta = target - self.V[state]
            self._N[state] += 1
            # V(s) = V(s) + (1/N(s)).(G(t) - V(s))
            self.V[state] += delta / self._N[state]
