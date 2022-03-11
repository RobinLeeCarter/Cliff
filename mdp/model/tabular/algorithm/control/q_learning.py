from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.tabular.agent.agent import Agent
from mdp import common
from mdp.model.tabular.algorithm.abstract.episodic_online_control import EpisodicOnlineControl


class QLearning(EpisodicOnlineControl):
    def __init__(self,
                 agent: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 name: str
                 ):
        super().__init__(agent, algorithm_parameters, name)
        self._alpha = self._algorithm_parameters.alpha
        self.title = f"{self.name} α={self._alpha}"
        self._create_q()

    def _do_training_step(self):
        ag = self._agent
        ag.choose_action()
        ag.take_action()

        target = ag.r + self._gamma * self.Q.max[ag.s]
        delta = target - self.Q[ag.prev_s, ag.prev_a]
        self.Q[ag.prev_s, ag.prev_a] += self._alpha * delta
        # update policy to be in-line with Q
        self._agent.policy[ag.prev_s] = self.Q.argmax[ag.prev_s]
