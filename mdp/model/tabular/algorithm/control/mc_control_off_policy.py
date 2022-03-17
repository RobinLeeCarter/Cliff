from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.tabular.agent.tabular_agent import TabularAgent
from mdp import common
from mdp.model.tabular.algorithm.abstract.episodic_monte_carlo import EpisodicMonteCarlo
from mdp.model.tabular.value_function.state_action_variable import StateActionVariable


class McControlOffPolicy(EpisodicMonteCarlo):
    def __init__(self,
                 agent: TabularAgent,
                 algorithm_parameters: common.AlgorithmParameters,
                 name: str
                 ):
        super().__init__(agent, algorithm_parameters, name)
        self._W: float = 1.0
        self._create_q()
        self._C = StateActionVariable(self._environment, initial_value=0.0)

    def initialize(self):
        super().initialize()
        self._set_target_policy_greedy_wrt_q()
        self._agent.behaviour_policy.refresh_policy_matrix()
        # self._environment.initialize_policy(self._agent.policy, self._policy_parameters)

    def _pre_process_episode(self):
        self._episode.generate_returns()
        self._W = 1.0

    def _process_time_step(self, t: int):
        s, a, target = self._episode.get_s_a_g(t)

        self._C[s, a] += self._W
        delta = target - self.Q[s, a]
        step_size = self._W / self._C[s, a]
        self.Q[s, a] += step_size * delta
        policy_a = self._agent.target_policy[s]
        best_a = self.Q.argmax[s]
        if best_a != policy_a:  # update policy only if different
            # get the agent to update the target policy since the behavioural policy might be linked and need to change
            self._agent.update_target_policy(s, best_a)
        if a != best_a:
            self._exit_episode = True
        else:
            behaviour_probability = self._agent.behaviour_policy.get_probability(s, a)
            self._W /= behaviour_probability
