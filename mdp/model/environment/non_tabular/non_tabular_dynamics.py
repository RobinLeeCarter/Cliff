from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABC

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.environment.non_tabular.non_tabular_environment import NonTabularEnvironment
from mdp.model.environment.dynamics import Dynamics


class NonTabularDynamics(Dynamics, ABC):
    def __init__(self, environment: NonTabularEnvironment, environment_parameters: common.EnvironmentParameters):
        """init top down"""
        super().__init__(environment, environment_parameters)
        self._environment: NonTabularEnvironment = environment
