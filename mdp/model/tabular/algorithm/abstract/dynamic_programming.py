from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Callable
import abc

if TYPE_CHECKING:
    from mdp.model.tabular.environment.tabular_environment import TabularEnvironment
    from mdp.model.tabular.agent.agent import Agent
    from mdp import common
from mdp.model.tabular.algorithm.abstract.algorithm import Algorithm


class DynamicProgramming(Algorithm, abc.ABC):
    def __init__(self,
                 environment: TabularEnvironment,
                 agent: Agent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(environment, agent, algorithm_parameters)
        self._theta = self._algorithm_parameters.theta
        self._iteration_timeout = self._algorithm_parameters.iteration_timeout

        # trainer callback
        self._step_callback: Optional[Callable[[], bool]] = None

    def set_step_callback(self, step_callback: Callable[[], bool]):
        self._step_callback = step_callback

    def initialize(self):
        super().initialize()
        self._environment.initialize_policy(self._agent.policy)

    @abc.abstractmethod
    def run(self):
        pass
