from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod


if TYPE_CHECKING:
    from mdp.model.non_tabular.agent.non_tabular_agent import NonTabularAgent
    from mdp.model.non_tabular.agent.non_tabular_episode import NonTabularEpisode
from mdp import common
from mdp.model.non_tabular.algorithm.abstract.nontabular_episodic import NonTabularEpisodic
from mdp.model.non_tabular.algorithm.batch_mixin.batch_feature_trajectories import BatchFeatureTrajectories
from mdp.model.non_tabular.feature.tile_coding.tile_coding import TileCoding


class NonTabularEpisodicOnline(NonTabularEpisodic, ABC):
    def __init__(self,
                 agent: NonTabularAgent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(agent, algorithm_parameters)
        self._alpha = self._algorithm_parameters.alpha

    @staticmethod
    def get_title(name: str, algorithm_parameters: common.AlgorithmParameters) -> str:
        return f"{name} α={algorithm_parameters.alpha}"

    def do_episode(self) -> NonTabularEpisode:
        self._agent.start_episode()
        # print(self._agent.state)
        self._start_episode()
        while (not self._agent.state.is_terminal)\
                and self._agent.t < self._episode_length_timeout\
                and self._agent.episode.cont:
            self._do_training_step()
            if self.batch_episodes == common.BatchEpisodes.FEATURE_TRAJECTORIES:
                assert isinstance(self, BatchFeatureTrajectories)
                self._append_feature_trajectory()
        self._end_episode()
        return self._agent.episode

    @abstractmethod
    def _do_training_step(self):
        pass

    def _update_parameters_based_on_feature(self):
        """update alpha based on number of tilings"""
        if isinstance(self._feature, TileCoding):
            self._alpha /= self._feature.tilings
