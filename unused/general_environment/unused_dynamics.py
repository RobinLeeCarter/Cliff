from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.environment.general.general_action import GeneralAction
    from mdp.model.environment.general.environment import Environment
from mdp.model.environment.general.general_state import GeneralState

Response = tuple[float, GeneralState]


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

    def get_start_state_distribution(self) -> common.Distribution[GeneralState]:
        """
        Starting state distribution
        If want to use something different to a Uniform list of States, override this method to return the distribution
        """
        start_states: list[GeneralState] = self.get_start_states()
        if start_states:
            if len(start_states) == 1:
                return common.SingularDistribution[GeneralState](start_states)
            else:
                return common.UniformMultinoulli[GeneralState](start_states)
        else:
            raise Exception("Empty list of start states so nowhere to start!")

    def get_start_states(self) -> list[GeneralState]:
        """If as simple as a list of start states then return it here and the distribution will be generated"""
        pass

    @abstractmethod
    def draw_response(self, state: GeneralState, action: GeneralAction) -> Response:
        """
        draw a single outcome for a single state and action
        """
