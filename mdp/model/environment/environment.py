from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from mdp.model.algorithm.abstract.algorithm import Algorithm
    from mdp.model.policy.policy import Policy
    from mdp.model.environment.dynamics import Dynamics

from mdp import common
from mdp.model.environment.state import State
from mdp.model.environment.action import Action


class Environment(ABC):
    """An abstract Environment for tabular or continuous cases"""
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        """
        :param environment_parameters: the parameters specific to the specific environment e.g. number_of_cars
        """
        self._environment_parameters = environment_parameters
        self.verbose: bool = environment_parameters.verbose
        # None to ensure not used when not used/initialised
        self.dynamics: Optional[Dynamics] = None

    @abstractmethod
    def build(self):
        """should always perform self.dynamics.build()"""
        pass

    # region Operation
    def _is_action_compatible_with_state(self, state: State, action: Action):
        # by default all actions are compatible with all states
        return True

    def initialize_policy(self, policy_: Policy, policy_parameters: common.PolicyParameters):
        pass

    @abstractmethod
    def from_state_perform_action(self, state: State, action: Action) -> tuple[float, State]:
        pass

    def update_grid_value_functions(self,
                                    algorithm: Algorithm,
                                    policy: Policy):
        pass

    def is_valued_state(self, state: State) -> bool:
        """Does the state have a valid value function V(s) or Q(s,a) e.g. unreachable states might not"""
        return not state.is_terminal
    # endregion
