from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Generic

from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.environment.general.general_environment import GeneralEnvironment

from mdp.model.environment.general.general_state import GeneralState
from mdp.model.environment.general.general_action import GeneralAction

State = TypeVar('State', bound=GeneralState)
Action = TypeVar('Action', bound=GeneralAction)


class GeneralDynamics(Generic[State, Action], ABC):
    def __init__(self, environment: GeneralEnvironment[State, Action],
                 environment_parameters: common.EnvironmentParameters):
        """init top down"""
        self._environment: GeneralEnvironment[State, Action] = environment
        self._environment_parameters: common.EnvironmentParameters = environment_parameters
        self._verbose: bool = environment_parameters.verbose
        self.is_built: bool = False

    def build(self):
        """build bottom up"""
        self.is_built = True

    @abstractmethod
    def draw_response(self, state: State, action: Action) -> tuple[float, State]:
        """
        draw a single outcome for a single state and action
        """

    # def get_start_state_distribution(self) -> common.Distribution[State]:
    #     """
    #     Starting state distribution
    #     If want to use something different to a Uniform list of States,
    #     override this method to return the distribution
    #     """
    #     start_states: list[State] = self.get_start_states()
    #     if start_states:
    #         if len(start_states) == 1:
    #             return common.SingularDistribution[State](start_states)
    #         else:
    #             return common.UniformMultinoulli[State](start_states)
    #     else:
    #         raise Exception("Empty list of start states so nowhere to start!")
    #
    # def get_start_states(self) -> list[State]:
    #     """If as simple as a list of start states then return it here and the distribution will be generated"""
    #     pass
