from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.tabular.agent.tabular_agent import TabularAgent
from mdp import common
from mdp.model.tabular.algorithm.abstract.episodic_monte_carlo import EpisodicMonteCarlo
from mdp.model.tabular.value_function.state_variable import StateVariable


class MCPredictionV(EpisodicMonteCarlo,
                    algorithm_type=common.AlgorithmType.MC_PREDICTION_V,
                    algorithm_name="MC Prediction (V)"):
    def __init__(self,
                 agent: TabularAgent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(agent, algorithm_parameters)
        self._create_v()
        self._N = StateVariable(self._environment, initial_value=0.0)

    @staticmethod
    def get_title(name: str, algorithm_parameters: common.AlgorithmParameters) -> str:
        return f"{name} first_visit={algorithm_parameters.first_visit}"

    def initialize(self):
        super().initialize()
        self._environment.initialize_policy(self._target_policy)

    def _process_time_step(self, t: int):
        # only do updates on the time-steps that should be done
        if (self._first_visit and self._episode.is_first_visit[t]) \
                or not self._first_visit:
            s = self._episode[t].s
            target = self._episode.G[t]
            delta = target - self.V[s]
            self._N[s] += 1
            # V(s) = V(s) + (1/N(s)).(G(t) - V(s))
            self.V[s] += delta / self._N[s]
