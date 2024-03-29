from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.base.environment.base_action import BaseAction
    from mdp.model.environment.general.environment import Environment
from mdp.model.base.environment.base_state import BaseState

Response = tuple[float, BaseState]


class Dynamics(ABC):
    def __init__(self, environment: Environment, environment_parameters: common.EnvironmentParameters):
        """init top down"""
        self._environment: Environment = environment
        self._environment_parameters: common.EnvironmentParameters = environment_parameters
        self._verbose: bool = environment_parameters.verbose
        self.is_built: bool = False

    def build(self):
        """build bottom up"""
        self.is_built = True

    def get_start_state_distribution(self) -> common.Distribution[BaseState]:
        """
        Starting state distribution
        If want to use something different to a Uniform list of States, override this method to return the distribution
        """
        start_states: list[BaseState] = self.get_start_states()
        if start_states:
            if len(start_states) == 1:
                return common.SingularDistribution[BaseState](start_states)
            else:
                return common.UniformMultinoulli[BaseState](start_states)
        else:
            raise Exception("Empty list of start states so nowhere to start!")

    def get_start_states(self) -> list[BaseState]:
        """If as simple as a list of start states then return it here and the distribution will be generated"""
        pass

    @abstractmethod
    def draw_response(self, state: BaseState, action: BaseAction) -> Response:
        """
        draw a single outcome for a single state and action
        """
