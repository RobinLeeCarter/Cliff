from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import abc

if TYPE_CHECKING:
    from mdp.model.tabular.environment.tabular_environment import TabularEnvironment
    from mdp.model.tabular.agent.agent import Agent
    from mdp.model.tabular.agent.episode import Episode
    from mdp import common
from mdp.model.tabular.algorithm.abstract.episodic import Episodic


class EpisodicMonteCarlo(Episodic, abc.ABC):
    def __init__(self,
                 environment_: TabularEnvironment,
                 agent: Agent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(environment_, agent, algorithm_parameters)
        self._episode: Optional[Episode] = None
        self._exit_episode: bool = False
        self._exploring_starts: bool = algorithm_parameters.exploring_starts

    @property
    def episode(self) -> Optional[Episode]:
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

    def process_episodes(self, episodes: list[Episode]):
        for episode in episodes:
            self.process_episode(episode)

    def process_episode(self, episode: Episode):
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

    @abc.abstractmethod
    def _process_time_step(self, t):
        pass
