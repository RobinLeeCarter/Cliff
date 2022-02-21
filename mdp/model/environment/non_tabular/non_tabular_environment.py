from __future__ import annotations
from typing import Optional, TYPE_CHECKING  # , Generic, TypeVar
from abc import ABC, abstractmethod

import numpy as np

if TYPE_CHECKING:
    from mdp.model.algorithm.value_function import state_function


from mdp import common
from mdp.model.environment.environment import Environment
from mdp.model.environment.non_tabular.non_tabular_state import NonTabularState
from mdp.model.environment.non_tabular.non_tabular_action import NonTabularAction
from mdp.model.environment.non_tabular.dims import Dims

# State = TypeVar('State', bound=NonTabularState)
# Action = TypeVar('Action', bound=NonTabularAction)


class NonTabularEnvironment(Environment, ABC):      # , Generic[State, Action]
    """An abstract Environment with continuous states but discrete actions"""
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        """
        :param environment_parameters: the parameters specific to the specific environment e.g. number_of_cars
        """
        super().__init__(environment_parameters)

        # action list and action lookup
        self.actions: list[NonTabularAction] = []
        self.actions_array: np.ndarray = np.empty(0, dtype=object)
        self.action_index: dict[NonTabularAction: int] = {}
        self._actions_always_compatible: bool = True

        self._last_state: Optional[NonTabularState] = None
        self._possible_actions_list: list[NonTabularAction] = []
        self._possible_actions_array: Optional[np.ndarray] = None

        # dimensions
        self._dims: Dims = Dims()

        # Distributions
        self._start_state_distribution: Optional[common.Distribution[NonTabularState]] = None

    def build(self):
        self._build_actions()
        self.action_index: dict[NonTabularAction, int] = {action: i for i, action in enumerate(self.actions)}
        self.actions_array: np.ndarray = np.array(self.actions)

        # defaults, and always used is
        self._possible_actions_list: list[NonTabularAction] = self.actions
        self._possible_actions_array: np.ndarray = np.ones(shape=(len(self.actions)), dtype=bool)

        self._build_dimensions()
        self._start_state_distribution: common.Distribution[NonTabularState] = self._get_start_state_distribution()

    @abstractmethod
    def _build_actions(self):
        pass

    @abstractmethod
    def _build_dimensions(self):
        pass

    @abstractmethod
    def _get_start_state_distribution(self) -> common.Distribution[NonTabularState]:
        pass

    # region Operation
    def build_possible_actions(self, state: NonTabularState, build_array: bool = True):
        if not self._actions_always_compatible:
            if state == self._last_state:   # this list at least is cached, maybe array too
                if build_array and self._possible_actions_array is None:    # needs array and array is not cached
                    self._possible_actions_array = np.array([self._is_action_compatible_with_state(state, action)
                                                             for action in self.actions], dtype=bool)
            else:
                if build_array:
                    self._possible_actions_array = np.array([self._is_action_compatible_with_state(state, action)
                                                             for action in self.actions], dtype=bool)
                    self._possible_actions_list = self.actions_array[self.possible_actions_array].tolist()
                else:
                    self._possible_actions_array = None
                    self._possible_actions_list = \
                        [action for action in self.actions if self._is_action_compatible_with_state(state, action)]
            self._last_state = state

    @property
    def actions_always_compatible(self) -> bool:
        return self._actions_always_compatible

    @property
    def possible_actions_list(self) -> list[NonTabularAction]:
        """list of just the possible actions"""
        return self._possible_actions_list

    @property
    def possible_actions_array(self) -> np.ndarray:
        """boolean array of the indexes of actions possible from the current state"""
        return self._possible_actions_array

    def draw_start_state(self) -> NonTabularState:
        return self._start_state_distribution.draw_one()

    def from_state_perform_action(self, state: NonTabularState, action: NonTabularAction) -> \
            tuple[float, NonTabularState]:
        if state.is_terminal:
            raise Exception("Environment: Trying to act in a terminal state.")
        if not self._is_action_compatible_with_state(state, action):
            raise Exception(f"_apply_action state {state} incompatible with action {action}")
        reward, new_state = self._draw_response(state, action)
        # if not new_state:
        #     new_state: State = self._start_state_distribution.draw_one()

        return reward, new_state

    @abstractmethod
    def _draw_response(self, state: NonTabularState, action: NonTabularAction) -> tuple[float, NonTabularState]:
        """
        draw a single outcome for a single state and action
        """

    # TODO: should StateFunction be more general e.g. Tabular vs Function Approximation
    def insert_state_function_into_graph3d(self,
                                           comparison: common.Comparison,
                                           v: state_function.StateFunction,
                                           parameter: Optional[any] = None):
        pass
    # endregion
