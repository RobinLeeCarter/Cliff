from __future__ import annotations
from typing import TYPE_CHECKING
import abc

if TYPE_CHECKING:
    from mdp.model.tabular.environment.tabular_environment import TabularEnvironment
    from mdp.model.tabular.agent.agent import Agent
    from mdp import common
from mdp.model.tabular.algorithm.abstract.episodic import Episodic


class EpisodicOnline(Episodic, abc.ABC):
    def __init__(self,
                 environment: TabularEnvironment,
                 agent: Agent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(environment, agent, algorithm_parameters)

    def do_episode(self, episode_length_timeout: int):
        self._agent.start_episode()
        self._start_episode()
        while (not self._agent.is_terminal)\
                and self._agent.t < episode_length_timeout\
                and self._agent.episode.cont:
            self._do_training_step()

    def _start_episode(self):
        pass

    @abc.abstractmethod
    def _do_training_step(self):
        pass

    # def _change_q_and_update_policy(self, s: int, a: int, value: float):
    #     self.Q[s, a] = value
    #     # update policy to be in-line with Q if it needs to change
    #     self._agent.policy[s] = self.Q.argmax[s]

        # if value > self.Q.max[s]:
        #     self._max_q[s] = value
        #     self._agent.policy[s] = a