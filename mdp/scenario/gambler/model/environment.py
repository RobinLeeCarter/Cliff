from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model.tabular.policy.tabular_policy import TabularPolicy
    from mdp.model.tabular.policy.deterministic import Deterministic
    from mdp.model.tabular.value_function import state_function

from mdp import common
from mdp.scenario.gambler.model.state import State
from mdp.scenario.gambler.model.action import Action
from mdp.scenario.gambler.comparison.environment_parameters import EnvironmentParameters
# from mdp.scenarios.gambler.grid_world import GridWorld
from mdp.scenario.gambler.model.dynamics import Dynamics

from mdp.model.tabular.environment.tabular_environment import TabularEnvironment


class Environment(TabularEnvironment[State, Action]):
    def __init__(self, environment_parameters: EnvironmentParameters):
        super().__init__(environment_parameters)
        self._environment_parameters: EnvironmentParameters = environment_parameters

        self._max_capital: int = environment_parameters.max_capital
        # dealer_card is x, player_sum is y : following the table in the book
        # grid_shape = (len(self._player_sums), len(self._dealers_cards))
        # self.grid_world: GridWorld = GridWorld(environment_parameters=environment_parameters, grid_shape=grid_shape)
        self.dynamics: Dynamics = Dynamics(environment=self, environment_parameters=environment_parameters)

    def _build_states(self):
        """set S"""
        # non-terminal states
        for capital in range(self._max_capital+1):
            is_terminal: bool = (capital == 0 or capital == self._max_capital)
            new_state: State = State(
                is_terminal=is_terminal,
                capital=capital
            )
            self.states.append(new_state)

    def _build_actions(self):
        for stake in range(0, self._max_capital):       # 0 stake is for graphs for terminal state
            new_action: Action = Action(
                stake=stake
            )
            self.actions.append(new_action)

    def _is_action_compatible_with_state(self, state: State, action: Action):
        if state.is_terminal:
            return action.stake == 0
        else:
            if 0 < action.stake <= state.capital and \
                    state.capital + action.stake <= self._max_capital:
                return True
            else:
                return False

    def initialize_policy(self, policy: TabularPolicy):
        hit: bool

        policy.zero_state_action()
        for s, state in enumerate(self.states):
            if state.is_terminal:
                initial_action = Action(stake=0)    # for graphs
            else:
                initial_action = Action(stake=1)
            policy.set_action(s, initial_action)

    def insert_state_function_into_graph2d(self,
                                           comparison: common.Comparison,
                                           v: state_function.StateFunction):
        x_list: list[int] = []
        y_list: list[float] = []
        for s, state in enumerate(self.states):
            if not state.is_terminal:
                x_list.append(state.capital)
                y_list.append(v[s])
                # print(state.capital, v[state])
        x_values = np.array(x_list, dtype=int)
        y_values = np.array(y_list, dtype=float)

        g = comparison.graph_values
        g.x_series = common.Series(title=g.x_label, values=x_values)
        g.graph_series = [common.Series(title=g.y_label, values=y_values)]
        g.show_graph = True
        g.title = "V(s)"
        g.x_label = "Capital"
        g.y_label = "V(s)"
        g.x_min = 0.0
        g.x_max = 100.0
        g.y_min = 0.0
        g.y_max = 1.0
        g.has_grid = True
        g.has_legend = False

    def insert_policy_into_graph2d(self,
                                   comparison: common.Comparison,
                                   policy: TabularPolicy):
        policy: Deterministic

        x_list: list[int] = []
        y_list: list[float] = []
        for s, state in enumerate(self.states):
            if not state.is_terminal:
                x_list.append(state.capital)
                action: Action = policy.get_action(s)   # type: ignore
                y_list.append(float(action.stake))
                # print(state.capital, v[state])
        x_values = np.array(x_list, dtype=int)
        y_values = np.array(y_list, dtype=float)

        g = comparison.graph_values
        g.x_series = common.Series(title=g.x_label, values=x_values)
        g.graph_series = [common.Series(title=g.y_label, values=y_values)]
        g.show_graph = True
        g.title = "Policy"
        g.x_label = "Capital"
        g.y_label = "Stake"
        g.x_min = 0.0
        g.x_max = 100.0
        g.y_min = 0.0
        g.y_max = None
        g.has_grid = True
        g.has_legend = False

    # def insert_state_function_into_graph3d(self,
    #                                        comparison: common.Comparison,
    #                                        v: state_function.StateFunction,
    #                                        parameter: Optional[any] = None):
    #     usable_ace: bool = parameter
    #     x_values = np.array(self._player_sums, dtype=int)
    #     y_values = np.array(self._dealers_cards, dtype=int)
    #     z_values = np.empty(shape=y_values.shape + x_values.shape, dtype=float)
    #
    #     for player_sum in self._player_sums:
    #         for dealers_card in self._dealers_cards:
    #             state: State = State(
    #                 is_terminal=False,
    #                 player_sum=player_sum,
    #                 usable_ace=usable_ace,
    #                 dealers_card=dealers_card,
    #             )
    #             x = player_sum - self._player_sum_min
    #             y = dealers_card - self._dealers_card_min
    #             z_values[y, x] = v[state]
    #             # print(player_sum, dealer_card, v[state])
    #
    #     g = comparison.graph3d_values
    #     if usable_ace:
    #         g.title = "Usable Ace"
    #     else:
    #         g.title = "No usable Ace"
    #     g.x_series = common.Series(title=g.x_label, values=x_values)
    #     g.y_series = common.Series(title=g.y_label, values=y_values)
    #     g.z_series = common.Series(title=g.z_label, values=z_values)

    # def update_grid_value_functions(self,
    #                                 algorithm_: algorithm.Algorithm,
    #                                 policy_: policy.Policy,
    #                                 parameter: any = None):
    #     usable_ace: bool = parameter
    #     # policy_: policy.Deterministic
    #     for state in self.states:
    #         if not state.is_terminal and state.usable_ace == usable_ace:
    #             # dealer_card is x, player_sum is y : following the table in the book
    #             x = state.dealers_card - self._dealers_card_min
    #             y = state.player_sum - self._player_sum_min
    #             position: common.XY = common.XY(x, y)
    #             action: Action = policy_[state]
    #             policy_value: int = int(action.hit)
    #             # print(position, transfer_1_to_2)
    #             self.grid_world.set_policy_value(
    #                 position=position,
    #                 policy_value=policy_value,
    #             )
