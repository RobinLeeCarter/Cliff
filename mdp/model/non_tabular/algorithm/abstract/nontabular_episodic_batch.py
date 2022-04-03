from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import numpy as np

if TYPE_CHECKING:
    from mdp.model.non_tabular.agent.non_tabular_agent import NonTabularAgent
    from mdp import common
# from mdp.model.non_tabular.feature.tile_coding.tile_coding import TileCoding
from mdp.model.non_tabular.algorithm.abstract.nontabular_episodic import NonTabularEpisodic


class NonTabularEpisodicBatch(NonTabularEpisodic, ABC):
    def __init__(self,
                 agent: NonTabularAgent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(agent, algorithm_parameters)
        self._alpha = self._algorithm_parameters.alpha

    @staticmethod
    def get_title(name: str, algorithm_parameters: common.AlgorithmParameters) -> str:
        return f"{name} α={algorithm_parameters.alpha}"

    def _update_parameters_based_on_feature(self):
        """update alpha based on number of tilings"""
        # if isinstance(self._feature, TileCoding):
        #     self._alpha /= self._feature.tilings

    def do_episode(self, episode_length_timeout: int):
        self._agent.start_episode()
        self._start_episode()
        while (not self._agent.state.is_terminal)\
                and self._agent.t < episode_length_timeout\
                and self._agent.episode.cont:
            self._do_training_step()

    def _start_episode(self):
        pass

    @abstractmethod
    def _do_training_step(self):
        pass

    def apply_delta_w_vectors(self, delta_w_vectors: list[np.ndarray]):
        delta_w_stack = np.stack(delta_w_vectors, axis=0)
        delta_w = np.sum(delta_w_stack, axis=0)
        self.Q.update_weights(delta_w)
