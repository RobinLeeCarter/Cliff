from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from mdp.model.non_tabular.agent.non_tabular_agent import NonTabularAgent
    from mdp import common
from mdp.model.non_tabular.algorithm.non_tabular_algorithm import NonTabularAlgorithm


class NonTabularEpisodic(NonTabularAlgorithm, ABC):
    def __init__(self,
                 agent: NonTabularAgent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(agent, algorithm_parameters)

    # TODO: should episode_length_timeout be passed in each time or be a property defaulted by algorithm_parameters
    # TODO: return episode?
    @abstractmethod
    def do_episode(self, episode_length_timeout: int):
        pass
