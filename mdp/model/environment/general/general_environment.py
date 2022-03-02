from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Generic, Optional
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from mdp.model.algorithm.abstract.algorithm import Algorithm
    from mdp.model.policy.policy import Policy
    from mdp.model.environment.general.general_dynamics import GeneralDynamics

from mdp import common
from mdp.model.environment.general.general_state import GeneralState
from mdp.model.environment.general.general_action import GeneralAction

State = TypeVar('State', bound=GeneralState)
Action = TypeVar('Action', bound=GeneralAction)


class GeneralEnvironment(Generic[State, Action], ABC):
    """An abstract Environment for tabular or continuous cases"""
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        """
        :param environment_parameters: the parameters specific to the specific environment e.g. number_of_cars
        """
        self._environment_parameters: common.EnvironmentParameters = environment_parameters
        self.verbose: bool = environment_parameters.verbose
        # None to ensure not used when not used/initialised

        # Distributions
        self._start_state_distribution: Optional[common.Distribution[State]] = None
        self._dynamics: Optional[GeneralDynamics[State, Action]] = None

    @abstractmethod
    def build(self):
        """call after init and could be slow"""
        pass

    # region Operation
    # noinspection PyUnusedLocal
    def _is_action_compatible_with_state(self, state: State, action: Action):
        # by default all actions are compatible with all states, override if neccesary
        return True

    # TODO: is this in the right place?
    def initialize_policy(self, policy_: Policy, policy_parameters: common.PolicyParameters):
        pass

    @abstractmethod
    def from_state_perform_action(self, state: State, action: Action) -> tuple[float, State]:
        pass

    # TODO: does this belong here?
    def update_grid_value_functions(self,
                                    algorithm: Algorithm,
                                    policy: Policy):
        pass

    # TODO: does this belong here or in the value_function?
    def is_valued_state(self, state: State) -> bool:
        """Does the state have a valid value function V(s) or Q(s,a) e.g. unreachable states might not"""
        return not state.is_terminal
    # endregion
