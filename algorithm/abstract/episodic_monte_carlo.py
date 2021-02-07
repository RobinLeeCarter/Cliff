import abc
from typing import Optional

import environment
import agent
from algorithm.abstract import episodic


class EpisodicMonteCarlo(episodic.Episodic, abc.ABC):
    name: str = "Error EpisodicOnline.name"

    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 verbose: bool = False
                 ):
        super().__init__(environment_, agent_, verbose)
        self.episode: Optional[agent.Episode] = None

    def do_episode(self, episode_length_timeout: int):
        self.agent.generate_episode()
        self.episode = self.agent.episode
        self._pre_process_episode()
        if self.episode.terminates and self.episode.T > 0:
            for t in range(self.episode.T - 1, -1, -1):     # T-1, T-2, ... 1, 0
                self._process_time_step(t)

    def _pre_process_episode(self):
        self.episode.generate_returns()

    @abc.abstractmethod
    def _process_time_step(self, t):
        pass
