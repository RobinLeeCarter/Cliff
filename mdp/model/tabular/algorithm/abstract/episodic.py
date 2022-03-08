from __future__ import annotations
from typing import TYPE_CHECKING
import abc

if TYPE_CHECKING:
    from mdp.model.tabular.environment.tabular_environment import TabularEnvironment
    from mdp.model.tabular.agent.agent import Agent
    from mdp import common
from mdp.model.tabular.algorithm.abstract.algorithm import Algorithm


class Episodic(Algorithm, abc.ABC):
    def __init__(self,
                 environment_: TabularEnvironment,
                 agent: Agent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(environment_, agent, algorithm_parameters)
        self.first_visit = self._algorithm_parameters.first_visit

    # TODO: should episode_length_timeout be passed in each time or be a property defaulted by algorithm_parameters
    @abc.abstractmethod
    def do_episode(self, episode_length_timeout: int):
        pass
