from __future__ import annotations
from typing import TYPE_CHECKING
import abc

if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment
    from mdp.model.agent.agent import Agent
    from mdp import common
from mdp.model.algorithm.abstract.algorithm import Algorithm


class Episodic(Algorithm, abc.ABC):
    def __init__(self,
                 environment_: Environment,
                 agent: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment_, agent, algorithm_parameters, policy_parameters)
        self.first_visit = self._algorithm_parameters.first_visit

    @abc.abstractmethod
    def do_episode(self, episode_length_timeout: int):
        pass
