from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.tabular.environment.tabular_environment import TabularEnvironment
    from mdp.model.tabular.agent.agent import Agent
from mdp import common
from mdp.model.tabular.algorithm.abstract.episodic_monte_carlo import EpisodicMonteCarlo
from mdp.model.tabular.algorithm.value_function.state_action_variable import StateActionVariable


class MCPredictionQ(EpisodicMonteCarlo):
    def __init__(self,
                 environment: TabularEnvironment,
                 agent: Agent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(environment, agent, algorithm_parameters)
        self._algorithm_type = common.AlgorithmType.MC_PREDICTION_Q
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name} first_visit={self.first_visit}"
        self._create_q()
        self._N = StateActionVariable(self._environment, initial_value=0.0)

    def initialize(self):
        super().initialize()
        self._environment.initialize_policy(self._agent.policy)

    def _process_time_step(self, t: int):
        # only do updates on the time-steps that should be done
        if (self.first_visit and self._episode.is_first_visit[t]) \
                or not self.first_visit:
            s = self._episode[t].s
            a = self._episode[t].a
            target = self._episode.G[t]
            delta = target - self.Q[s, a]
            self._N[s, a] += 1
            # Q(s,a) = Q(s,a) + (1/N(s,a)).(G(t) - Q(s,a))
            self.Q[s, a] += delta / self._N[s, a]
