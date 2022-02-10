from __future__ import annotations
from typing import TYPE_CHECKING
import abc

if TYPE_CHECKING:
    from mdp.model.environment.tabular.tabular_environment import TabularEnvironment
    from mdp.model.agent.agent import Agent
    from mdp import common
from mdp.model.algorithm.abstract.episodic import Episodic


class EpisodicOnline(Episodic, abc.ABC):
    def __init__(self,
                 environment_: TabularEnvironment,
                 agent: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment_, agent, algorithm_parameters, policy_parameters)

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
