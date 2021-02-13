from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import abc

if TYPE_CHECKING:
    import environment
    import agent
from algorithm.abstract import episodic


class EpisodicMonteCarlo(episodic.Episodic, abc.ABC):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 algorithm_parameters: dict[str, any]
                 ):
        super().__init__(environment_, agent_, algorithm_parameters)
        self.episode: Optional[agent.Episode] = None

    def do_episode(self, episode_length_timeout: int):
        self.agent.generate_episode(episode_length_timeout)
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
