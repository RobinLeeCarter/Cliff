from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.tabular.environment.tabular_environment import TabularEnvironment
    from mdp.model.tabular.agent.agent import Agent
from mdp import common
from mdp.model.tabular.algorithm.abstract.episodic_online_control import EpisodicOnlineControl


class Sarsa(EpisodicOnlineControl):
    def __init__(self,
                 environment: TabularEnvironment,
                 agent: Agent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(environment, agent, algorithm_parameters)
        self._alpha = self._algorithm_parameters.alpha
        self._algorithm_type = common.AlgorithmType.SARSA
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name} α={self._alpha}"
        self._create_q()

    def _start_episode(self):
        self._agent.choose_action()

    def _do_training_step(self):
        ag = self._agent
        ag.take_action()
        ag.choose_action()

        target: float = ag.r + self._gamma * self.Q[ag.s, ag.a]
        delta: float = target - self.Q[ag.prev_s, ag.prev_a]
        # print(f"a: {ag.a}"
        #       f"\tQ[curr]: {self.Q[ag.s, ag.a]}"
        #       f"\tQ[prev]: {self.Q[ag.prev_s, ag.prev_a]}"
        #       f"\tdelta: {delta}"
        #       f"\talpha: {self._alpha}")
        self.Q[ag.prev_s, ag.prev_a] += self._alpha * delta
        ag.policy[ag.prev_s] = self.Q.argmax[ag.prev_s]

        # previous verison: update policy to be in-line with Q by recalculation every time
        # ag.policy[ag.prev_s] = self.Q.argmax_over_actions(ag.prev_s)
