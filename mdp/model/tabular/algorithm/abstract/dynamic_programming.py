from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Callable
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from mdp.model.tabular.agent.agent import Agent
    from mdp import common
from mdp.model.tabular.algorithm.tabular_algorithm import TabularAlgorithm


class DynamicProgramming(TabularAlgorithm, ABC):
    def __init__(self,
                 agent: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 name: str
                 ):
        super().__init__(agent, algorithm_parameters, name)
        self._theta = self._algorithm_parameters.theta
        self.title = f"{self.name} θ={self._theta}"
        self._iteration_timeout = self._algorithm_parameters.iteration_timeout

        # trainer callback
        self._step_callback: Optional[Callable[[], bool]] = None

    def set_step_callback(self, step_callback: Callable[[], bool]):
        self._step_callback = step_callback

    def initialize(self):
        super().initialize()
        self._environment.initialize_policy(self._agent.policy)

    @abstractmethod
    def run(self):
        pass
