from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from mdp.model.tabular.agent.tabular_agent import TabularAgent
    from mdp.model.tabular.agent.tabular_episode import TabularEpisode
    from mdp import common
from mdp.model.tabular.algorithm.tabular_algorithm import TabularAlgorithm


class TabularEpisodic(TabularAlgorithm, ABC,
                      episodic=True):
    def __init__(self,
                 agent: TabularAgent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(agent, algorithm_parameters)
        self._first_visit: bool = self._algorithm_parameters.first_visit
        self._episode_length_timeout: int = 1

    def apply_settings(self, settings: common.Settings):
        self._episode_length_timeout = settings.episode_length_timeout
        super().apply_settings(settings)

    @abstractmethod
    def do_episode(self) -> TabularEpisode:
        pass
