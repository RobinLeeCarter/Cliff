from __future__ import annotations
from typing import TYPE_CHECKING
import abc

if TYPE_CHECKING:
    from mdp.model import environment, agent
    from mdp import common
from mdp.model.algorithm.abstract import algorithm_


class DynamicProgramming(algorithm_.Algorithm, abc.ABC):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment_, agent_, algorithm_parameters, policy_parameters)
        self._theta = self._algorithm_parameters.theta
        self._iteration_timeout = self._algorithm_parameters.iteration_timeout
        self._dynamics: environment.Dynamics = self._environment.dynamics
        assert self._dynamics is not None

    def initialize(self):
        self._environment.initialize_policy(self._agent.policy, self._policy_parameters)
        super().initialize()

    @abc.abstractmethod
    def run(self):
        pass