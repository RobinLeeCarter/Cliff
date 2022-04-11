from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np

from mdp.model.non_tabular.algorithm.abstract.nontabular_episodic_online import NonTabularEpisodicOnline


class NonTabularEpisodicBatch(NonTabularEpisodicOnline, ABC,
                              batch_episodes=True):
    # start of episodes
    @abstractmethod
    def start_episodes(self):
        pass

    # end of episodes
    def get_delta_weights(self) -> np.ndarray:
        pass

    # end of batch single-processing
    def apply_episodes(self):
        """for use with batch episodes but a single process"""

    # end of batch multiprocessing
    def apply_delta_w_vectors(self, delta_w_vectors: list[np.ndarray]):
        delta_w_stack = np.stack(delta_w_vectors, axis=0)
        # has to be average otherwise it's going to move far too far, especially at the start
        delta_w = np.average(delta_w_stack, axis=0)
        # print(f"{np.count_nonzero(delta_w)=}")
        self.apply_delta_w_vector(delta_w)

    def apply_delta_w_vector(self, delta_w: np.ndarray):
        """depends on the specific implementation of weights (in theory)"""
        pass
        # self.Q.update_weights(delta_w)

    # def unpack_delta_w_vectors(self, delta_w_vectors: list[np.ndarray]) -> np.ndarray:
    #     delta_w_stack = np.stack(delta_w_vectors, axis=0)
    #     delta_w = np.sum(delta_w_stack, axis=0)
    #     return delta_w
