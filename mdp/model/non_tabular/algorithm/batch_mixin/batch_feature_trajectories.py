from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.non_tabular.agent.non_tabular_agent import NonTabularAgent
    from mdp.model.non_tabular.agent.reward_feature_vector import FeatureTrajectory
from mdp import common
from mdp.model.non_tabular.algorithm.batch_mixin.batch__episodic import BatchEpisodic


class BatchFeatureTrajectories(BatchEpisodic, ABC,
                               batch_episodes=common.BatchEpisodes.FEATURE_TRAJECTORIES):
    def __init__(self,
                 agent: NonTabularAgent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(agent, algorithm_parameters)
        self._feature_trajectories: list[FeatureTrajectory] = []

    @property
    def feature_trajectories(self) -> list[FeatureTrajectory]:
        return self._feature_trajectories

    # start of episodes
    def start_episodes(self):
        self._feature_trajectories = []

    # feature trajectory processing
    def add_feature_trajectories(self, feature_trajectories: list[FeatureTrajectory]):
        self._feature_trajectories.extend(feature_trajectories)

    # end of batch single-processing
    def apply_feature_trajectories(self):
        for feature_trajectory in self._feature_trajectories:
            self._apply_feature_trajectory(feature_trajectory)

    @abstractmethod
    def _apply_feature_trajectory(self, feature_trajectory: FeatureTrajectory):
        pass
