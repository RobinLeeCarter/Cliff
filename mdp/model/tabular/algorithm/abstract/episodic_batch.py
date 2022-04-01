from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from mdp.model.tabular.agent.tabular_agent import TabularAgent
    from mdp.model.tabular.agent.tabular_episode import TabularEpisode
    from mdp import common
from mdp.model.tabular.algorithm.abstract.tabular_episodic import TabularEpisodic


# TODO: Is this a good idea
class EpisodicBatch(TabularEpisodic, ABC):
    def __init__(self,
                 agent: TabularAgent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(agent, algorithm_parameters)
        self._episode: Optional[TabularEpisode] = None
        self._exit_episode: bool = False
        self._exploring_starts: bool = algorithm_parameters.exploring_starts

    @property
    def episode(self) -> Optional[TabularEpisode]:
        return self._episode

    def do_episode(self, episode_length_timeout: int):
        episode = self._agent.generate_episode(episode_length_timeout, self._exploring_starts)
        self.process_episode(episode)
        # self._episode = self._agent.generate_episode(episode_length_timeout, self._exploring_starts)
        # self._pre_process_episode()
        # self._exit_episode = False
        # if self._episode.terminates and self._episode.T > 0:
        #     for t in range(self._episode.T - 1, -1, -1):     # T-1, T-2, ... 1, 0
        #         self._process_time_step(t)
        #         if self._exit_episode:
        #             break

    def process_episodes(self, episodes: list[TabularEpisode]):
        for episode in episodes:
            self.process_episode(episode)

    def process_episode(self, episode: TabularEpisode):
        self._episode = episode
        self._pre_process_episode()
        self._exit_episode = False
        if self._episode.terminates and self._episode.T > 0:
            for t in range(self._episode.T - 1, -1, -1):     # T-1, T-2, ... 1, 0
                self._process_time_step(t)
                if self._exit_episode:
                    break

    def _pre_process_episode(self):
        self._episode.generate_returns()

    @abstractmethod
    def _process_time_step(self, t):
        pass
