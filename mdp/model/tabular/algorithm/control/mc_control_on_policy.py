from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.tabular.agent.tabular_agent import TabularAgent
from mdp import common
from mdp.model.tabular.algorithm.abstract.episodic_monte_carlo import EpisodicMonteCarlo
from mdp.model.tabular.value_function.state_action_variable import StateActionVariable


class McControlOnPolicy(EpisodicMonteCarlo):
    def __init__(self,
                 agent: TabularAgent,
                 algorithm_parameters: common.AlgorithmParameters,
                 name: str
                 ):
        super().__init__(agent, algorithm_parameters, name)
        self._create_q()
        self._N = StateActionVariable(self._environment, initial_value=0.0)

    def initialize(self):
        super().initialize()
        self._environment.initialize_policy(self._target_policy)

    def _process_time_step(self, t: int):
        # only do updates on the time-steps that should be done
        if (self.first_visit and self._episode.is_first_visit[t]) \
                or not self.first_visit:
            s, a, target = self._episode.get_s_a_g(t)
            delta = target - self.Q[s, a]
            self._N[s, a] += 1.0
            # Q(s,a) = Q(s,a) + (1/N(s,a)).(G(t) - Q(s,a))
            self.Q[s, a] += delta / self._N[s, a]
            self._target_policy[s] = self.Q.argmax[s]
