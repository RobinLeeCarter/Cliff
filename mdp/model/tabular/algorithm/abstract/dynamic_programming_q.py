from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC

if TYPE_CHECKING:
    from mdp.model.tabular.agent.agent import Agent
    from mdp import common
from mdp.model.tabular.algorithm.abstract.dynamic_programming import DynamicProgramming


class DynamicProgrammingQ(DynamicProgramming, ABC):
    def __init__(self,
                 agent: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 name: str
                 ):
        super().__init__(agent, algorithm_parameters, name)
        self._create_q()
