from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np


class BatchMixin(ABC):
    # start of episodes
    @abstractmethod
    def start_episodes(self):
        pass

    # end of episodes
    def get_delta_weights(self) -> np.ndarray:
        pass

    # end of batch
    def apply_delta_w_vectors(self, delta_w_vectors: list[np.ndarray]):
        delta_w_stack = np.stack(delta_w_vectors, axis=0)
        delta_w = np.sum(delta_w_stack, axis=0)
        self.apply_delta_w_vector(delta_w)

    @abstractmethod
    def apply_delta_w_vector(self, delta_w: np.ndarray):
        """depends on the specific implementation of weights (in theory)"""
        pass
        # self.Q.update_weights(delta_w)

    # def unpack_delta_w_vectors(self, delta_w_vectors: list[np.ndarray]) -> np.ndarray:
    #     delta_w_stack = np.stack(delta_w_vectors, axis=0)
    #     delta_w = np.sum(delta_w_stack, axis=0)
    #     return delta_w
