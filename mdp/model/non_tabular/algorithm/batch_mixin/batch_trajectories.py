from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.non_tabular.agent.non_tabular_agent import NonTabularAgent
    from mdp.model.non_tabular.agent.reward_state_action import Trajectory
from mdp import common
from mdp.model.non_tabular.algorithm.batch_mixin.batch__episodic import BatchEpisodic


class BatchTrajectories(BatchEpisodic, ABC,
                        batch_episodes=common.BatchEpisodes.TRAJECTORIES):
    def __init__(self,
                 agent: NonTabularAgent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(agent, algorithm_parameters)
        self._trajectories: list[Trajectory] = []

    @property
    def trajectories(self) -> list[Trajectory]:
        return self._trajectories

    # start of episodes
    def start_episodes(self):
        self._trajectories = []

    def _end_episode(self):
        if self._agent.state.is_terminal:
            self._trajectories.append(self._agent.episode.trajectory)

    # trajectory processing
    def add_trajectories(self, trajectories: list[Trajectory]):
        self._trajectories.extend(trajectories)

    # end of batch single-processing
    def apply_trajectories(self):
        for trajectory in self._trajectories:
            self._apply_trajectory(trajectory)

    @abstractmethod
    def _apply_trajectory(self, trajectory: Trajectory):
        pass
