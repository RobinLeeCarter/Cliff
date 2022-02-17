from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from mdp.model.algorithm.value_function import state_function


from mdp import common
from mdp.model.environment.environment import Environment
from mdp.model.environment.state import State
from mdp.model.environment.action import Action
from mdp.model.environment.non_tabular.dims import Dims


class NonTabularEnvironment(Environment, ABC):
    """An abstract Environment with continuous states but discrete actions"""
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        """
        :param environment_parameters: the parameters specific to the specific environment e.g. number_of_cars
        """
        super().__init__(environment_parameters)

        # action list and action lookup
        self.actions: list[Action] = []
        self.action_index: dict[Action: int] = {}

        # dimensions
        self._dims: Dims = Dims()

        # Distributions
        self._start_state_distribution: Optional[common.Distribution[State]] = None

    def build(self):
        self._build_actions()
        self.action_index = {action: i for i, action in enumerate(self.actions)}
        self._build_dimensions()
        self._start_state_distribution = self._get_start_state_distribution()

    @abstractmethod
    def _build_actions(self):
        pass

    @abstractmethod
    def _build_dimensions(self):
        pass

    @abstractmethod
    def _get_start_state_distribution(self) -> common.Distribution[State]:
        pass

    # region Operation
    @abstractmethod
    def _draw_response(self, state: State, action: Action) -> tuple[float, State]:
        """
        draw a single outcome for a single state and action
        """

    def draw_start_state(self) -> State:
        return self._start_state_distribution.draw_one()

    def from_state_perform_action(self, state: State, action: Action) -> tuple[float, State]:
        if state.is_terminal:
            raise Exception("Environment: Trying to act in a terminal state.")
        if not self._is_action_compatible_with_state(state, action):
            raise Exception(f"_apply_action state {state} incompatible with action {action}")
        reward, new_state = self._draw_response(state, action)
        # if not new_state:
        #     new_state: State = self._start_state_distribution.draw_one()

        return reward, new_state

    # TODO: should StateFunction be more general e.g. Tabular vs Function Approximation
    def insert_state_function_into_graph3d(self,
                                           comparison: common.Comparison,
                                           v: state_function.StateFunction,
                                           parameter: Optional[any] = None):
        pass
    # endregion