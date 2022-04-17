from __future__ import annotations

from mdp import common
from mdp.model.non_tabular.algorithm.episodic.sarsa.sarsa_feature_trajectories import SarsaFeatureTrajectories


class SarsaFeatureTrajectoriesParallel(SarsaFeatureTrajectories,
                                       algorithm_type=common.AlgorithmType.NT_SARSA_FEATURE_TRAJECTORIES_PARALLEL,
                                       algorithm_name="Sarsa Parallel Feature Trajectories"):
    pass
