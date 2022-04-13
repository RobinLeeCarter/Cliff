from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from mdp.model.non_tabular.agent.non_tabular_agent import NonTabularAgent
    from mdp.model.non_tabular.agent.non_tabular_episode import NonTabularEpisode
    from mdp import common
from mdp.model.non_tabular.algorithm.non_tabular_algorithm import NonTabularAlgorithm


class NonTabularEpisodic(NonTabularAlgorithm, ABC,
                         episodic=True):
    def __init__(self,
                 agent: NonTabularAgent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(agent, algorithm_parameters)
        self._episode_length_timeout: int = 1

    def apply_settings(self, settings: common.Settings):
        self._episode_length_timeout = settings.episode_length_timeout
        super().apply_settings(settings)

    @abstractmethod
    def do_episode(self) -> NonTabularEpisode:
        pass
