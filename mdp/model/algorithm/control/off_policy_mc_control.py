from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment
    from mdp.model.agent.agent import Agent
from mdp import common
from mdp.model.algorithm.abstract.episodic_monte_carlo import EpisodicMonteCarlo
from mdp.model.algorithm.value_function.state_action_variable import StateActionVariable


class OffPolicyMcControl(EpisodicMonteCarlo):
    def __init__(self,
                 environment_: Environment,
                 agent_: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment_, agent_, algorithm_parameters, policy_parameters)
        self._algorithm_type = common.AlgorithmType.OFF_POLICY_MC_CONTROL
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name}"
        self._W: float = 1.0
        self._create_q()
        self._C = StateActionVariable(self._environment, initial_value=0.0)

    def initialize(self):
        super().initialize()
        self._set_policy_greedy_wrt_q()

    def _pre_process_episode(self):
        self._episode.generate_returns()
        self._W = 1.0

    def _process_time_step(self, t: int):
        s = self._episode[t].s
        a = self._episode[t].a

        self._C[s, a] += self._W
        target = self._episode.G[t]
        delta = target - self.Q[s, a]
        step_size = self._W / self._C[s, a]
        self.Q[s, a] += step_size * delta
        best_a = self.Q.argmax[s]
        self._agent.target_policy[s] = best_a
        if a != best_a:
            self._exit_episode = True
        else:
            behaviour_probability = self._agent.behaviour_policy.get_probability(s, a)
            self._W /= behaviour_probability
