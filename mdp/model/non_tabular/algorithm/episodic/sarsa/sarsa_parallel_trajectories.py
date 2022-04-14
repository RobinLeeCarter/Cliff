from __future__ import annotations

from mdp.model.non_tabular.algorithm.episodic.sarsa.sarsa import Sarsa
from mdp.model.non_tabular.algorithm.batch_mixin.batch_trajectories import BatchTrajectories
from mdp.model.non_tabular.algorithm.episodic.sarsa.sarsa_apply_trajectory_mixin import SarsaApplyTrajectoryMixin
from mdp import common


class SarsaParallelTrajectories(Sarsa, BatchTrajectories, SarsaApplyTrajectoryMixin,
                                algorithm_type=common.AlgorithmType.NT_SARSA_PARALLEL_TRAJECTORIES,
                                algorithm_name="Episodic Sarsa Parallel Episodes"):
    pass
