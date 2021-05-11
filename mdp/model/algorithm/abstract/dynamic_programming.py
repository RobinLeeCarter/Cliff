from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Callable
import abc

if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment
    from mdp.model.environment.dynamics import Dynamics
    from mdp.model.agent.agent import Agent
    from mdp import common
from mdp.model.algorithm.abstract.algorithm import Algorithm


class DynamicProgramming(Algorithm, abc.ABC):
    def __init__(self,
                 environment_: Environment,
                 agent_: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment_, agent_, algorithm_parameters, policy_parameters)
        self._theta = self._algorithm_parameters.theta
        self._iteration_timeout = self._algorithm_parameters.iteration_timeout
        self._dynamics: Dynamics = self._environment.dynamics
        self._iteration: int = 0

        # trainer callback
        self._step_callback: Optional[Callable[[], bool]] = None

    def set_step_callback(self, step_callback: Callable[[], bool]):
        self._step_callback = step_callback

    def initialize(self):
        super().initialize()
        self._environment.initialize_policy(self._agent.policy, self._policy_parameters)
        self._iteration = 0

    @abc.abstractmethod
    def run(self):
        pass
