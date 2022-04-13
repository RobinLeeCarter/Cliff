from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod


if TYPE_CHECKING:
    from mdp.model.tabular.agent.tabular_agent import TabularAgent
    from mdp.model.tabular.agent.tabular_episode import TabularEpisode
    from mdp import common
from mdp.model.tabular.algorithm.abstract.tabular_episodic import TabularEpisodic


class TabularEpisodicOnline(TabularEpisodic, ABC):
    def __init__(self,
                 agent: TabularAgent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(agent, algorithm_parameters)

    @staticmethod
    def get_title(name: str, algorithm_parameters: common.AlgorithmParameters) -> str:
        return f"{name} α={algorithm_parameters.alpha}"

    def do_episode(self) -> TabularEpisode:
        self._agent.start_episode()
        self._start_episode()
        while (not self._agent.is_terminal)\
                and self._agent.t < self._episode_length_timeout\
                and self._agent.episode.cont:
            self._do_training_step()
        return self._agent.episode

    def _start_episode(self):
        pass

    @abstractmethod
    def _do_training_step(self):
        pass

    # def _change_q_and_update_policy(self, s: int, a: int, value: float):
    #     self.Q[s, a] = value
    #     # update policy to be in-line with Q if it needs to change
    #     self._target_policy[s] = self.Q.argmax[s]

        # if value > self.Q.max[s]:
        #     self._max_q[s] = value
        #     self._target_policy[s] = a
