from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

import numpy as np

if TYPE_CHECKING:
    from mdp.model.algorithm.abstract.algorithm import Algorithm
    from mdp.model.policy.policy import Policy
    from mdp.model.algorithm.value_function import state_function

from mdp import common
from mdp.model.environment.state import State
from mdp.model.environment.action import Action
from mdp.model.environment.dynamics import Dynamics

S_A = tuple[int, int]


class Environment(ABC):
    """A GridWorld Environment - too hard to make general at this point"""
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        self._environment_parameters = environment_parameters
        self.verbose: bool = environment_parameters.verbose

        # state and states
        self.states: list[State] = []
        self.state_index: dict[State: int] = {}

        # action and actions
        self.actions: list[Action] = []
        self.action_index: dict[Action: int] = {}
        self.is_terminal: list[bool] = []

        # almost all interactions with environment must be using state and action
        # exception boolean array of whether a in A(s) for a given [s, a]
        # possibly should be part of agent to enforce API but should be able to have mutliple agents for one evironment
        self.s_a_compatibility: np.ndarray = np.empty(0, dtype=bool)
        self.compatible_s_a: list[S_A] = []                 # for rapid access
        self.possible_actions: np.ndarray = np.empty(0, dtype=int)
        self.one_over_possible_actions: np.ndarray = np.empty(0, dtype=float)

        # Distributions
        self.s_a_distribution: Optional[common.UniformDistribution[S_A]] = None
        self.start_s_distribution: Optional[common.UniformDistribution[int]] = None

        # None to ensure not used when not used/initialised
        self.dynamics: Optional[Dynamics] = None

    def build(self):
        self._build_states()
        self.state_index = {state: i for i, state in enumerate(self.states)}
        self._build_actions()
        self.action_index = {action: i for i, action in enumerate(self.actions)}
        self._build_state_actions()
        self._build_helper_arrays()
        self.dynamics.build()
        self._build_distributions()

    def state_action_index(self, state: State, action: Action) -> S_A:
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
        self.s_a_compatibility = np.zeros(shape=(len(self.states), len(self.actions)), dtype=bool)
        for s, state in enumerate(self.states):
            if not state.is_terminal:
                for a, action in enumerate(self.actions):
                    if self._is_action_compatible_with_state(state, action):
                        self.s_a_compatibility[s, a] = True
                        self.compatible_s_a.append((s, a))

    def _is_action_compatible_with_state(self, state: State, action: Action):
        return True

    def _build_helper_arrays(self):
        self.is_terminal = [state.is_terminal for state in self.states]
        # self.one_over_possible_actions = np.zeros(shape=(len(self.states)), dtype=float)
        self.possible_actions = np.count_nonzero(self.s_a_compatibility, axis=1).astype(dtype=float)
        self.one_over_possible_actions = np.zeros_like(self.possible_actions)
        non_zero = (self.possible_actions != 0.0)
        np.reciprocal(self.possible_actions, out=self.one_over_possible_actions, where=non_zero)

    def _build_distributions(self):
        self.s_a_distribution = common.UniformDistribution[S_A](self.compatible_s_a)

        start_states = self.dynamics.get_start_states()
        start_s = [self.state_index[state] for state in start_states]
        if len(start_s) == 1:
            self.start_s_distribution = common.SingularDistribution[int](start_s)
        else:
            self.start_s_distribution = common.UniformDistribution[int](start_s)

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

    def initialize_policy(self, policy_: Policy, policy_parameters: common.PolicyParameters):
        pass

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

    def from_state_perform_action(self, state: State, action: Action) -> tuple[float, State]:
        s = self.state_index[state]
        a = self.action_index[action]
        if state.is_terminal:
            raise Exception("Environment: Trying to act in a terminal state.")
        if not self.s_a_compatibility[s, a]:
            raise Exception(f"_apply_action state {state} incompatible with action {action}")
        reward, new_state = self.dynamics.draw_response(state, action)
        if not new_state:
            new_s = self.start_s_distribution.draw_one()
            new_state: State = self.states[new_s]

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

    def update_grid_value_functions(self,
                                    algorithm: Algorithm,
                                    policy: Policy):
        pass

    def is_valued_state(self, state: State) -> bool:
        return False

    def output_mode(self):
        pass
    # endregion
