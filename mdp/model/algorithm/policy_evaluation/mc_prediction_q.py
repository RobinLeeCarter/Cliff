from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment
    from mdp.model.agent.agent import Agent
from mdp import common
from mdp.model.algorithm.abstract.episodic_monte_carlo import EpisodicMonteCarlo
from mdp.model.algorithm.value_function.state_action_variable import StateActionVariable


class MCPredictionQ(EpisodicMonteCarlo):
    def __init__(self,
                 environment_: Environment,
                 agent_: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment_, agent_, algorithm_parameters, policy_parameters)
        self._algorithm_type = common.AlgorithmType.MC_PREDICTION_Q
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name} first_visit={self.first_visit}"
        self._create_q()
        self._N = StateActionVariable(self._environment, initial_value=0.0)

    def initialize(self):
        super().initialize()
        self._environment.initialize_policy(self._agent.policy, self._policy_parameters)

    def _process_time_step(self, t: int):
        # only do updates on the time-steps that should be done
        if (self.first_visit and self._episode.is_first_visit[t]) \
                or not self.first_visit:
            state = self._episode[t].state
            action = self._episode[t].action
            target = self._episode.G[t]
            delta = target - self.Q[state, action]
            self._N[state, action] += 1
            # Q(s,a) = Q(s,a) + (1/N(s,a)).(G(t) - Q(s,a))
            self.Q[state, action] += delta / self._N[state, action]
