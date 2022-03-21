from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC

if TYPE_CHECKING:
    from mdp.model.base.agent.base_agent import BaseAgent
    from mdp import common


class BaseAlgorithm(ABC):
    def __init__(self,
                 agent: BaseAgent,
                 algorithm_parameters: common.AlgorithmParameters,
                 name: str
                 ):
        self._agent: BaseAgent = agent
        # self._environment: GeneralEnvironment = self._agent.environment
        self._algorithm_parameters: common.AlgorithmParameters = algorithm_parameters
        self._verbose = self._algorithm_parameters.verbose

        self.name: str = name
        self.title: str = self.get_title(name, algorithm_parameters)

        self._gamma: float = self._agent.gamma

    def __repr__(self):
        return f"{self.title}"

    def initialize(self):
        pass

    def parameter_changes(self, iteration: int):
        pass

    # noinspection PyUnusedLocal
    @staticmethod
    def get_title(name: str, algorithm_parameters: common.AlgorithmParameters) -> str:
        return name
