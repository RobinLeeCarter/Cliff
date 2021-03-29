from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment
    from mdp.model.agent.agent import Agent
from mdp import common
from mdp.model.algorithm import abstract, value_function


class OffPolicyMcControl(abstract.EpisodicMonteCarlo):
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
        self._C = value_function.StateActionVariable(self._environment, initial_value=0.0)

    def initialize(self):
        super().initialize()
        self._make_policy_greedy_wrt_q()

    def _pre_process_episode(self):
        self._episode.generate_returns()
        self._W = 1.0

    def _process_time_step(self, t: int):
        state = self._episode[t].state
        action = self._episode[t].action

        self._C[state, action] += self._W
        target = self._episode.G[t]
        delta = target - self.Q[state, action]
        step_size = self._W / self._C[state, action]
        self.Q[state, action] += step_size * delta
        best_action = self.Q.argmax_over_actions(state)
        self._agent.target_policy[state] = best_action
        if action != best_action:
            self._exit_episode = True
        else:
            behaviour_probability = self._agent.behaviour_policy.get_probability(state, action)
            self._W /= behaviour_probability
