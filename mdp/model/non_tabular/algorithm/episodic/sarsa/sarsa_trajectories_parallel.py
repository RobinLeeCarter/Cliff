from __future__ import annotations

from mdp.model.non_tabular.algorithm.episodic.sarsa.sarsa_trajectories import SarsaTrajectories
from mdp import common


class SarsaTrajectoriesParallel(SarsaTrajectories,
                                algorithm_type=common.AlgorithmType.NT_SARSA_TRAJECTORIES_PARALLEL,
                                algorithm_name="Episodic Sarsa Parallel Trajectories"):
    pass
