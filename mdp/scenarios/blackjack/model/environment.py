from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model.policy.policy import Policy
    from mdp.model.algorithm.value_function import state_function

from mdp import common
from mdp.model.environment import environment

from mdp.scenarios.blackjack.model.state import State
from mdp.scenarios.blackjack.model.action import Action
from mdp.scenarios.blackjack.model.environment_parameters import EnvironmentParameters
from mdp.scenarios.blackjack.model.grid_world import GridWorld
from mdp.scenarios.blackjack.model.dynamics import Dynamics


class Environment(environment.Environment):
    def __init__(self, environment_parameters: EnvironmentParameters):

        super().__init__(environment_parameters)

        # super().__init__(environment_parameters_, grid_world_)

        # downcast states and actions so properties can be used freely
        self.states: list[State] = self.states
        self.actions: list[Action] = self.actions
        self._state: State = self._state
        self._action: Action = self._action

        self._player_sum_min = 11
        self._player_sum_max = 21
        self._dealers_card_min = 1
        self._dealers_card_max = 10
        self._player_sums = [x for x in range(self._player_sum_min, self._player_sum_max+1)]
        self._dealers_cards = [x for x in range(self._dealers_card_min, self._dealers_card_max+1)]

        # dealer_card is x, player_sum is y : following the table in the book
        grid_shape = (len(self._player_sums), len(self._dealers_cards))
        self.grid_world: GridWorld = GridWorld(environment_parameters=environment_parameters, grid_shape=grid_shape)
        self.dynamics: Dynamics = Dynamics(environment_=self, environment_parameters=environment_parameters)

    # region Sets
    def _build_states(self):
        """set S"""
        # non-terminal states
        for player_sum in self._player_sums:
            for usable_ace in [False, True]:
                for dealers_card in self._dealers_cards:
                    new_state: State = State(
                        is_terminal=False,
                        player_sum=player_sum,
                        usable_ace=usable_ace,
                        dealers_card=dealers_card,
                    )
                    self.states.append(new_state)

        # terminal states
        for result in [-1, 0, +1]:
            new_state: State = State(
                is_terminal=True,
                result=result
            )
            self.states.append(new_state)

    def _build_actions(self):
        for hit in [False, True]:
            new_action: Action = Action(
                hit=hit
            )
            self.actions.append(new_action)

    def is_action_compatible_with_state(self, state_: State, action_: Action):
        if state_.player_sum == 21 and action_.hit:
            return False
        else:
            return True
    # endregion

    # region Operation
    def initialize_policy(self, policy_: Policy, policy_parameters: common.PolicyParameters):
        hit: bool
        for state in self.states:
            # don't add a policy for terminal states at all
            if not state.is_terminal:
                if state.player_sum >= 20:
                    hit = False
                else:
                    hit = True
                initial_action = Action(hit)
                policy_[state] = initial_action

    def insert_state_function_into_graph3d_ace(self,
                                               comparison: common.Comparison,
                                               v: state_function.StateFunction,
                                               usable_ace: bool):
        x_values = np.array(self._player_sums, dtype=int)
        y_values = np.array(self._dealers_cards, dtype=int)
        z_values = np.empty(shape=y_values.shape + x_values.shape, dtype=float)

        for player_sum in self._player_sums:
            for dealers_card in self._dealers_cards:
                state: State = State(
                    is_terminal=False,
                    player_sum=player_sum,
                    usable_ace=usable_ace,
                    dealers_card=dealers_card,
                )
                x = player_sum - self._player_sum_min
                y = dealers_card - self._dealers_card_min
                z_values[y, x] = v[state]
                # print(player_sum, dealer_card, v[state])

        g = comparison.graph3d_values
        if usable_ace:
            g.title = "Usable Ace"
        else:
            g.title = "No usable Ace"
        g.x_series = common.Series(title=g.x_label, values=x_values)
        g.y_series = common.Series(title=g.y_label, values=y_values)
        g.z_series = common.Series(title=g.z_label, values=z_values)

    def update_grid_policy_ace(self, policy: Policy, usable_ace: bool):
        # policy_: policy.Deterministic
        for state in self.states:
            if not state.is_terminal and state.usable_ace == usable_ace:
                # dealer_card is x, player_sum is y : following the table in the book
                x = state.dealers_card - self._dealers_card_min
                y = state.player_sum - self._player_sum_min
                position: common.XY = common.XY(x, y)
                action: Action = policy[state]
                policy_value: int = int(action.hit)
                # print(position, transfer_1_to_2)
                self.grid_world.set_policy_value(
                    position=position,
                    policy_value=policy_value,
                )

    def is_valued_state(self, state: State) -> bool:
        return not state.is_terminal
    # endregion
