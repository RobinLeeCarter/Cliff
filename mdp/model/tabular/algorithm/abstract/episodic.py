from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from mdp.model.tabular.agent.agent import Agent
    from mdp import common
from mdp.model.tabular.algorithm.tabular_algorithm import TabularAlgorithm


class Episodic(TabularAlgorithm, ABC):
    def __init__(self,
                 agent: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 name: str
                 ):
        super().__init__(agent, algorithm_parameters, name)
        self.first_visit = self._algorithm_parameters.first_visit

    # TODO: should episode_length_timeout be passed in each time or be a property defaulted by algorithm_parameters
    @abstractmethod
    def do_episode(self, episode_length_timeout: int):
        pass
