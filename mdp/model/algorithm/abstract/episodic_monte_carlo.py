from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import abc

if TYPE_CHECKING:
    from mdp.model import environment, agent
    from mdp import common
from mdp.model.algorithm.abstract import episodic_


class EpisodicMonteCarlo(episodic_.Episodic, abc.ABC):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(environment_, agent_, algorithm_parameters)
        self._episode: Optional[agent.Episode] = None
        self._exit_episode: bool = False

    @property
    def episode(self) -> Optional[agent.Episode]:
        return self._episode

    def do_episode(self, episode_length_timeout: int):
        self._episode = self._agent.generate_episode(episode_length_timeout)
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
