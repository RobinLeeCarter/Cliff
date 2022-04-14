from __future__ import annotations

import numpy as np

from mdp.model.non_tabular.algorithm.batch_mixin.batch_delta_weights import BatchDeltaWeights
from mdp.model.non_tabular.algorithm.episodic.sarsa.sarsa import Sarsa
from mdp import common


class SarsaParallelDeltaWeights(Sarsa, BatchDeltaWeights,
                                algorithm_type=common.AlgorithmType.NT_SARSA_PARALLEL_DELTA_W,
                                algorithm_name="Sarsa Parallel Delta Weights"):
    def start_episodes(self):
        # have Q make a copy so can compare later to calculate the delta
        self.Q.copy_w()

    def get_delta_weights(self) -> np.ndarray:
        return self.Q.get_delta_weights()

    def apply_delta_w_vector(self, delta_w: np.ndarray):
        self.Q.update_weights(delta_w)
