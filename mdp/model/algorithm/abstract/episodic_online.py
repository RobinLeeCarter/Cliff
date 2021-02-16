from __future__ import annotations
from typing import TYPE_CHECKING
import abc

if TYPE_CHECKING:
    from mdp.model import environment, agent
    import common
from mdp.model.algorithm.abstract import episodic


class EpisodicOnline(episodic.Episodic, abc.ABC):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 ):
        super().__init__(environment_, agent_, algorithm_parameters)

    def do_episode(self, episode_length_timeout: int):
        self._agent.start_episode()
        self._start_episode()
        while (not self._agent.state.is_terminal) and self._agent.t < episode_length_timeout:
            self._do_training_step()

    def _start_episode(self):
        pass

    @abc.abstractmethod
    def _do_training_step(self):
        pass
