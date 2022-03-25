from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from mdp.model.tabular.policy.tabular_policy import TabularPolicy

from mdp import common
from mdp.model.base.environment.base_state import BaseState
from mdp.model.base.environment.base_action import BaseAction


class Environment(ABC):
    """An abstract Environment for tabular or continuous cases"""
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        """
        :param environment_parameters: the parameters specific to the specific environment e.g. number_of_cars
        """
        self._environment_parameters = environment_parameters
        self.verbose: bool = environment_parameters.verbose
        # None to ensure not used when not used/initialised

    @abstractmethod
    def build(self):
        """call after init and could be slow"""
        pass

    # region Operation
    def _is_action_compatible_with_state(self, state: BaseState, action: BaseAction):
        # by default all actions are compatible with all states
        return True

    def initialize_policy(self, policy_: TabularPolicy, policy_parameters: common.PolicyParameters):
        pass

    @abstractmethod
    def from_state_perform_action(self, state: BaseState, action: BaseAction) -> tuple[float, BaseState]:
        pass

    def update_grid_value_functions(self,
                                    algorithm: Algorithm,
                                    policy: TabularPolicy):
        pass

    def is_valued_state(self, state: BaseState) -> bool:
        """Does the state have a valid value function V(s) or Q(s,a) e.g. unreachable states might not"""
        return not state.is_terminal
    # endregion
