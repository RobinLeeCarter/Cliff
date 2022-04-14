from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model.non_tabular.agent.non_tabular_agent import NonTabularAgent
from mdp import common
from mdp.model.non_tabular.algorithm.batch_mixin.batch__episodic import BatchEpisodic


class BatchDeltaWeights(BatchEpisodic, ABC,
                        batch_episodes=common.BatchEpisodes.DELTA_WEIGHTS):
    def __init__(self,
                 agent: NonTabularAgent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(agent, algorithm_parameters)

    # end of episodes
    @abstractmethod
    def get_delta_weights(self) -> np.ndarray:
        pass

    # end of batch multiprocessing
    def apply_delta_w_vectors(self, delta_w_vectors: list[np.ndarray]):
        delta_w_stack = np.stack(delta_w_vectors, axis=0)
        # has to be average otherwise it's going to move far too far, especially at the start
        delta_w = np.average(delta_w_stack, axis=0)
        # print(f"{np.count_nonzero(delta_w)=}")
        self.apply_delta_w_vector(delta_w)

    @abstractmethod
    def apply_delta_w_vector(self, delta_w: np.ndarray):
        """depends on the specific implementation of weights (in theory)"""
        pass
        # self.Q.update_weights(delta_w)

    # def unpack_delta_w_vectors(self, delta_w_vectors: list[np.ndarray]) -> np.ndarray:
    #     delta_w_stack = np.stack(delta_w_vectors, axis=0)
    #     delta_w = np.average(delta_w_stack, axis=0)
    #     return delta_w
