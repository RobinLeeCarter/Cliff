from __future__ import annotations
from typing import TYPE_CHECKING

# import numpy as np

if TYPE_CHECKING:
    # from mdp.model.algorithm.abstract.algorithm import Algorithm
    from mdp.model.policy.policy import Policy
    from mdp.model.algorithm.value_function import state_function

from mdp import common
from mdp.model.environment.non_tabular.non_tabular_environment import NonTabularEnvironment
from mdp.model.environment.non_tabular.dimension.float_dimension import FloatDimension
from mdp.model.environment.non_tabular.dimension.category_dimension import CategoryDimension

# from mdp.scenarios.mountain_car.model.state import State
from mdp.scenarios.mountain_car.model.action import Action
from mdp.scenarios.mountain_car.model.environment_parameters import EnvironmentParameters
from mdp.scenarios.mountain_car.model.dynamics import Dynamics


class Environment(NonTabularEnvironment):
    def __init__(self, environment_parameters: EnvironmentParameters):

        super().__init__(environment_parameters)

        # downcast states and actions so properties can be used freely
        self.actions: list[Action] = self.actions

        self.position_dimension = FloatDimension(min=-1.2, max=0.5)
        self.velocity_dimension = FloatDimension(min=-0.07, max=0.07)

        self.dynamics: Dynamics = Dynamics(environment_=self, environment_parameters=environment_parameters)

    # region Sets
    def _build_actions(self):
        self.actions = [
            Action(acceleration=-1.0),
            Action(acceleration=0.0),
            Action(acceleration=1.0)
        ]

    def _build_dimensions(self):
        self.float_dimensions = [self.position_dimension, self.velocity_dimension]
        action_dimension = CategoryDimension(possible_values=len(self.actions))
        self.category_dimensions = [action_dimension]
    # endregion

    # region Operation
    def initialize_policy(self, policy: Policy, policy_parameters: common.PolicyParameters):
        pass
        # hit: bool
        #
        # policy.zero_state_action()
        # for s, state in enumerate(self.states):
        #     # don't add an action to the policy for terminal states at all
        #     if not state.is_terminal:
        #         if state.player_sum >= 20:
        #             hit = False
        #         else:
        #             hit = True
        #         initial_action: Action = Action(hit)
        #         policy.set_action(s, initial_action)

    def insert_state_function_into_graph3d_ace(self,
                                               comparison: common.Comparison,
                                               v: state_function.StateFunction,
                                               usable_ace: bool):
        pass
        # x_values = np.array(self._player_sums, dtype=int)
        # y_values = np.array(self._dealers_cards, dtype=int)
        # z_values = np.empty(shape=y_values.shape + x_values.shape, dtype=float)
        #
        # for player_sum in self._player_sums:
        #     for dealers_card in self._dealers_cards:
        #         state: State = State(
        #             is_terminal=False,
        #             player_sum=player_sum,
        #             usable_ace=usable_ace,
        #             dealers_card=dealers_card,
        #         )
        #         x = player_sum - self._player_sum_min
        #         y = dealers_card - self._dealers_card_min
        #         s = self.state_index[state]
        #         z_values[y, x] = v[s]
        #         # print(player_sum, dealer_card, v[state])
        #
        # g = comparison.graph3d_values
        # if usable_ace:
        #     g.title = "Usable Ace"
        # else:
        #     g.title = "No usable Ace"
        # g.x_series = common.Series(title=g.x_label, values=x_values)
        # g.y_series = common.Series(title=g.y_label, values=y_values)
        # g.z_series = common.Series(title=g.z_label, values=z_values)
    # endregion
