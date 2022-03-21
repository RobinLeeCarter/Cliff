from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC

if TYPE_CHECKING:
    # from mdp.model.non_tabular.environment.non_tabular_environment import NonTabularEnvironment
    from mdp.model.non_tabular.agent.non_tabular_agent import NonTabularAgent
    from mdp import common

from mdp.model.base.algorithm.base_algorithm import BaseAlgorithm


class NonTabularAlgorithm(BaseAlgorithm, ABC):
    def __init__(self,
                 agent: NonTabularAgent,
                 algorithm_parameters: common.AlgorithmParameters,
                 name: str
                 ):
        super().__init__(agent, algorithm_parameters, name)
        self._agent: NonTabularAgent = agent
        # self._environment: NonTabularEnvironment = self._agent.environment

