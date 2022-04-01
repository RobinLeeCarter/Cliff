from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC


if TYPE_CHECKING:
    from mdp.model.non_tabular.agent.non_tabular_agent import NonTabularAgent
    from mdp import common

from mdp.model.non_tabular.algorithm.non_tabular_algorithm import NonTabularAlgorithm


class SemiGradientTd0(NonTabularAlgorithm, ABC):
    # TODO: complete
    def __init__(self,
                 agent: NonTabularAgent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(agent, algorithm_parameters)
        # self._environment: NonTabularEnvironment = self._agent.environment

    def apply_result(self, result: common.Result):
        raise Exception("apply_result not implemented")
