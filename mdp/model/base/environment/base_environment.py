from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Generic
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from mdp.model.base.algorithm.base_algorithm import BaseAlgorithm
    from mdp.model.base.policy.base_policy import BasePolicy

from mdp import common
from mdp.model.base.environment.base_state import BaseState
from mdp.model.base.environment.base_action import BaseAction

State = TypeVar('State', bound=BaseState)
Action = TypeVar('Action', bound=BaseAction)


class BaseEnvironment(Generic[State, Action], ABC):
    """An abstract Environment for tabular or continuous cases"""
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        """
        :param environment_parameters: the parameters specific to the specific environment e.g. number_of_cars
        """
        self._environment_parameters: common.EnvironmentParameters = environment_parameters
        self.verbose: bool = environment_parameters.verbose

    @abstractmethod
    def build(self):
        """call after init and could be slow"""

    # region Operation
    # noinspection PyUnusedLocal
    def _is_action_compatible_with_state(self, state: State, action: Action) -> bool:
        # by default all actions are compatible with all states, override if neccesary
        return True

    # TODO: is this in the right place?
    def initialize_policy(self, policy: BasePolicy):
        pass

    @abstractmethod
    def from_state_perform_action(self, state: State, action: BaseAction) -> tuple[float, BaseState]:
        pass

    # TODO: does this belong here?
    def update_grid_value_functions(self,
                                    algorithm: BaseAlgorithm):
        pass

    # TODO: does this belong here or in the value_function?
    def is_valued_state(self, state: BaseState) -> bool:
        """Does the state have a valid value function V(s) or Q(s,a) e.g. unreachable states might not"""
        return not state.is_terminal
    # endregion
