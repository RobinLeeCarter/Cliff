from __future__ import annotations

# import numpy as np

from mdp.model.non_tabular.algorithm.batch_mixin.batch_shared_weights import BatchSharedWeights
from mdp.model.non_tabular.algorithm.episodic.sarsa.sarsa import Sarsa
from mdp import common


class SarsaSharedWeightsParallel(Sarsa, BatchSharedWeights,
                                 algorithm_type=common.AlgorithmType.NT_SARSA_SHARED_WEIGHTS_PARALLEL,
                                 algorithm_name="Sarsa Parallel Shared Weights"):
    pass
