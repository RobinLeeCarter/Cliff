from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Generic

from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.base.environment.base_environment import BaseEnvironment

from mdp.model.base.environment.base_state import BaseState
from mdp.model.base.environment.base_action import BaseAction

State = TypeVar('State', bound=BaseState)
Action = TypeVar('Action', bound=BaseAction)


class GeneralDynamics(Generic[State, Action], ABC):
    def __init__(self, environment: BaseEnvironment[BaseState, BaseAction],
                 environment_parameters: common.EnvironmentParameters):
        """init top down"""
        self._environment: BaseEnvironment[BaseState, BaseAction] = environment
        self._environment_parameters: common.EnvironmentParameters = environment_parameters
        self._verbose: bool = environment_parameters.verbose
        self.is_built: bool = False

    @abstractmethod
    def draw_response(self, state: BaseState, action: BaseAction) -> tuple[float, BaseState]:
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
