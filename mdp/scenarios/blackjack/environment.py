from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model import algorithm, policy
    from mdp.model.algorithm.value_function import state_function

from mdp import common
from mdp.model import environment

from mdp.scenarios.blackjack.state import State
from mdp.scenarios.blackjack.action import Action
from mdp.scenarios.blackjack.environment_parameters import EnvironmentParameters
# from mdp.scenarios.blackjack.grid_world import GridWorld
from mdp.scenarios.blackjack.dynamics import Dynamics


class Environment(environment.Environment):
    def __init__(self, environment_parameters: EnvironmentParameters):

        super().__init__(environment_parameters)

        # super().__init__(environment_parameters_, grid_world_)

        # downcast states and actions so properties can be used freely
        self.states: list[State] = self.states
        self.actions: list[Action] = self.actions
        self._state: State = self._state
        self._action: Action = self._action

        # self.grid_world: GridWorld = GridWorld(environment_parameters)
        self.dynamics: Dynamics = Dynamics(environment_=self, environment_parameters=environment_parameters)

        self._max_cars: int = environment_parameters.max_cars
        self._max_transfers: int = environment_parameters.max_transfers

    # region Sets
    def _build_states(self):
        """set S"""
        # non-terminal states
        for player_sum in range(12, 21+1):
            for usable_ace in [False, True]:
                for dealers_card in range(1, 10+1):
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
        return True
    # endregion

    # region Operation
    def initialize_policy(self, policy_: policy.Policy, policy_parameters: common.PolicyParameters):
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

    # def insert_state_function_into_graph3d(self, comparison: common.Comparison, v: state_function.StateFunction):
    #     x_values = np.arange(self._max_cars + 1, dtype=float)
    #     y_values = np.arange(self._max_cars + 1, dtype=float)
    #     z_values = np.empty(shape=(self._max_cars + 1, self._max_cars + 1), dtype=float)
    #
    #     for cars1 in range(self._max_cars+1):
    #         for cars2 in range(self._max_cars+1):
    #             state: State = State(
    #                 ending_cars_1=cars1,
    #                 ending_cars_2=cars2,
    #                 is_terminal=False,
    #             )
    #             z_values[cars2, cars1] = v[state]
    #             # print(cars1, cars2, v[state])
    #
    #     g = comparison.graph3d_values
    #     g.x_series = common.Series(title=g.x_label, values=x_values)
    #     g.y_series = common.Series(title=g.y_label, values=y_values)
    #     g.z_series = common.Series(title=g.z_label, values=z_values)

    # def update_grid_value_functions(self, algorithm_: algorithm.Algorithm, policy_: policy.Policy):
    #     # policy_: policy.Deterministic
    #     for state in self.states:
    #         position: common.XY = common.XY(x=state.ending_cars_2, y=state.ending_cars_1)     # reversed like in book
    #         action: Action = policy_[state]
    #         transfer_1_to_2: int = action.transfer_1_to_2
    #         # print(position, transfer_1_to_2)
    #         self.grid_world.set_policy_value(
    #             position=position,
    #             policy_value=transfer_1_to_2,
    #         )
    #         # if algorithm_.Q:
    #         #     policy_action: Optional[environment.Action] = policy_[state]
    #         #     policy_action: Action
    #         #     policy_move: Optional[common.XY] = None
    #         #     if policy_action:
    #         #         policy_move = policy_action.move
    #         #     for action_ in self.actions_for_state(state):
    #         #         is_policy: bool = (policy_move and policy_move == action_.move)
    #         #         self.grid_world.set_state_action_function(
    #         #             position=state.position,
    #         #             move=action_.move,
    #         #             q_value=algorithm_.Q[state, action_],
    #         #             is_policy=is_policy
    #         #         )
    #     # print(self.grid_world.output_squares)

    def is_valued_state(self, state: State) -> bool:
        return not state.is_terminal
    # endregion
