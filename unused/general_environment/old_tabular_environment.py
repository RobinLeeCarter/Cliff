from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

import numpy as np

if TYPE_CHECKING:
    from mdp.model.tabular.value_function import state_function
    from mdp.model.tabular.environment.tabular_dynamics import TabularDynamics

from mdp import common
from mdp.model.environment.general.environment import Environment
from mdp.model.general.environment.general_state import GeneralState
from mdp.model.general.environment.general_action import GeneralAction

S_A = tuple[int, int]


class TabularEnvironment(Environment, ABC):
    """An abstract Environment with discrete tabular states and actions"""
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        """
        :param environment_parameters: the parameters specific to the specific environment e.g. number_of_cars
        """
        super().__init__(environment_parameters)
        self.dynamics: Optional[TabularDynamics] = None

        # state and states
        self.states: list[GeneralState] = []               # ordered list of full States
        self.state_index: dict[GeneralState, int] = {}     # lookup from full State to state_index (s)
        self.is_terminal: list[bool] = []           # bool: is a given state_index (s) a terminal state?

        # action and actions
        self.actions: list[GeneralAction] = []
        self.action_index: dict[GeneralAction, int] = {}

        # almost all interactions with environment must be using state and action
        # exception boolean array of whether a in A(s) for a given [s, a]
        # possibly should be part of agent to enforce API but should be able to have mutliple agents for one evironment

        # bool of whether (s, a) combination is valid (action can be taken in state)
        self.s_a_compatibility: np.ndarray = np.empty(0, dtype=bool)
        # list of compatible (s, a) combinations to loop over efficiently
        self.compatible_s_a: list[S_A] = []                 # for rapid access
        # number of possible actions for each state
        self.possible_actions: np.ndarray = np.empty(0, dtype=int)
        # 1 / self.possible_actions for each state
        self.one_over_possible_actions: np.ndarray = np.empty(0, dtype=float)

        # Distributions
        self.s_a_distribution: Optional[common.DiscreteDistribution[S_A]] = None
        self.start_s_distribution: Optional[common.DiscreteDistribution[int]] = None

    def build(self):
        self._build_states()
        self.state_index = {state: i for i, state in enumerate(self.states)}
        self._build_actions()
        self.action_index = {action: i for i, action in enumerate(self.actions)}
        self._build_state_actions()
        self._build_helper_arrays()
        self.dynamics.build()
        self._build_distributions()

    def state_action_index(self, state: GeneralState, action: GeneralAction) -> S_A:
        state_index = self.state_index[state]
        action_index = self.action_index[action]
        return state_index, action_index

    # region Sets
    @abstractmethod
    def _build_states(self):
        pass

    @abstractmethod
    def _build_actions(self):
        pass

    def _build_state_actions(self):
        """materialise A(s)"""
        # TODO: Faster if invert this (assume normally compatible)
        self.s_a_compatibility = np.zeros(shape=(len(self.states), len(self.actions)), dtype=bool)
        for s, state in enumerate(self.states):
            if not state.is_terminal:
                for a, action in enumerate(self.actions):
                    if self._is_action_compatible_with_state(state, action):
                        self.s_a_compatibility[s, a] = True
                        self.compatible_s_a.append((s, a))

    def _build_helper_arrays(self):
        self.is_terminal = [state.is_terminal for state in self.states]
        # self.one_over_possible_actions = np.zeros(shape=(len(self.states)), dtype=float)
        self.possible_actions = np.count_nonzero(self.s_a_compatibility, axis=1).astype(dtype=float)
        non_zero: np.ndarray = (self.possible_actions != 0.0)
        self.one_over_possible_actions = np.zeros_like(self.possible_actions)
        np.reciprocal(self.possible_actions, out=self.one_over_possible_actions, where=non_zero)

    def _build_distributions(self):
        self.s_a_distribution = common.UniformMultinoulli[S_A](self.compatible_s_a)

        start_states = self.dynamics.get_start_states()
        start_s = [self.state_index[state] for state in start_states]
        if len(start_s) == 1:
            self.start_s_distribution = common.SingularDistribution[int](start_s)
        else:
            self.start_s_distribution = common.UniformMultinoulli[int](start_s)

    # endregion

    # region Operation
    # def get_random_state_action(self) -> tuple[State, Action]:
    #     state = random.choice([state for state in self.states if not state.is_terminal])
    #     action = random.choice(self.actions_for_state[state])
    #     return state, action

    # @profile
    # def get_random_s_a(self) -> S_A:
    #     return self.s_a_distribution.draw_one()
        # flat = np.flatnonzero(self.s_a_compatibility)
        # choice = np.random.choice(flat)
        # s, a = np.unravel_index(choice, self.s_a_compatibility.shape)
        # return s, self.is_terminal[s], a
        # choice = utils.n_choice(len(self.compatible_s_a))
        # s, a = self.compatible_s_a[choice]
        # return s, self.is_terminal[s], a

    def insert_state_function_into_graph3d(self,
                                           comparison: common.Comparison,
                                           v: state_function.StateFunction,
                                           parameter: Optional[any] = None):
        pass

    # def start_s(self) -> int:
    #     state = self.dynamics.get_a_start_state()
    #     return self.state_index[state]

    # def start_s(self) -> int:
    #     state = self._get_a_start_state()
    #     s: int = self.state_index[state]
    #     return s

    # def from_state_perform_action(self, state: State, action: Action) -> Response:
    #     if state.is_terminal:
    #         raise Exception("Environment: Trying to act in a terminal state.")
    #     if not self.is_action_compatible_with_state(state, action):
    #         raise Exception(f"_apply_action state {state} incompatible with action {action}")
    #     response: Response = self.dynamics.draw_response(state, action)
    #     return response

    def from_state_perform_action(self, state: GeneralState, action: GeneralAction) -> tuple[float, GeneralState]:
        s = self.state_index[state]
        a = self.action_index[action]
        if state.is_terminal:
            raise Exception("Environment: Trying to act in a terminal state.")
        if not self.s_a_compatibility[s, a]:
            raise Exception(f"_apply_action state {state} incompatible with action {action}")
        reward, new_state = self.dynamics.draw_response(state, action)
        if not new_state:
            new_s = self.start_s_distribution.draw_one()
            new_state: GeneralState = self.states[new_s]

        return reward, new_state

    # @profile
    def from_s_perform_a(self, s: int, a: int) -> tuple[float, int, bool]:

        state = self.states[s]
        if state.is_terminal:
            raise Exception("Environment: Trying to act in a terminal state.")

        if a == -1:
            action = None
        else:
            action = self.actions[a]
            if not self.s_a_compatibility[s, a]:
                raise Exception(f"_apply_action state {state} incompatible with action {action}")

        reward, new_state = self.dynamics.draw_response(state, action)
        if new_state:
            new_s = self.state_index[new_state]
            is_terminal = new_state.is_terminal
        else:
            new_s = self.start_s_distribution.draw_one()
            is_terminal = self.is_terminal[new_s]

        return reward, new_s, is_terminal
    # endregion
