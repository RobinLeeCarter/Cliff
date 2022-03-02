from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # from mdp.model.algorithm.abstract.algorithm import Algorithm
    from mdp.model.policy.policy import Policy
    from mdp.model.algorithm.value_function import state_function
    # from mdp.model.environment.non_tabular.non_tabular_state import NonTabularState
    # from mdp.model.environment.non_tabular.non_tabular_action import NonTabularAction

from mdp import common
from mdp.model.environment.non_tabular.non_tabular_environment import NonTabularEnvironment
from mdp.model.environment.non_tabular.dimension.float_dimension import FloatDimension
from mdp.model.environment.non_tabular.dimension.category_dimension import CategoryDimension

from mdp.scenarios.mountain_car.model.state import State
from mdp.scenarios.mountain_car.model.action import Action
from mdp.scenarios.mountain_car.model.dynamics import Dynamics
from mdp.scenarios.mountain_car.model.environment_parameters import EnvironmentParameters
from mdp.scenarios.mountain_car.model.start_state_distribution import StartStateDistribution
from mdp.scenarios.mountain_car.enums import Dim


class Environment(NonTabularEnvironment[State, Action, StartStateDistribution, Dynamics]):
    def __init__(self, environment_parameters: EnvironmentParameters):
        super().__init__(environment_parameters, actions_always_compatible=True)

        # self._dynamics = Dynamics(self, environment_parameters)
        print(type(self._start_state_distribution))
        print(type(self._dynamics))

    def _build_actions(self):
        self.actions = [
            Action(acceleration=-1.0),
            Action(acceleration=0.0),
            Action(acceleration=1.0)
        ]

    def _build_dimensions(self):
        # insertion order is critical
        self._dims.state_float[Dim.POSITION] = FloatDimension(min=-1.2, max=0.5)
        self._dims.state_float[Dim.VELOCITY] = FloatDimension(min=-0.07, max=0.07)
        self._dims.action_category[Dim.ACCELERATION] = CategoryDimension(possible_values=len(self.actions))

        # self.float_dimensions = [self._position_dimension, self._velocity_dimension]
        # action_dimension = CategoryDimension(possible_values=len(self.actions))
        # self.category_dimensions = [action_dimension]

    def _build_start_state_distribution(self) -> StartStateDistribution:
        return StartStateDistribution(self._dims)

    def _build_dynamics(self) -> Dynamics:
        return Dynamics(self, self._environment_parameters)

    def mountain(self):
        print("^")

    # def _set_start_state_distribution(self):
    #     self._start_state_distribution: StartStateDistribution = StartStateDistribution(self._dims)

    # region Operation
    # def draw_start_state(self) -> State:
    #     return super().draw_start_state()   # type: ignore

        # state: NonTabularState = super().draw_start_state()
        # state: State
        # return state

        # state = super().draw_start_state()
        # if isinstance(state, State):
        #     return state

    def _draw_response(self, state: State, action: Action) -> tuple[float, State]:
        position_dim = self._dims.state_float[Dim.POSITION]
        velocity_dim = self._dims.state_float[Dim.VELOCITY]
        new_position: float
        new_velocity: float
        is_terminal: bool = False
        reward: float = -1.0

        # environment rules from Sutton and Barto RL 10.1 p245
        projected_position = state.position + state.velocity
        if projected_position < position_dim.min:
            new_position = position_dim.min
            new_velocity = 0.0
        elif projected_position > position_dim.max:
            new_position = projected_position
            new_velocity = 0.0
            is_terminal = True
            reward = 0.0
        else:
            new_position = projected_position
            # ẋ(t) + 0.001*A(t) - 0.0025*cos( 3 * x(t) )
            projected_velocity = state.velocity + 0.001 * action.acceleration - 0.0025 * math.cos(3.0 * state.position)
            new_velocity = velocity_dim.bound(projected_velocity)
        new_state = State(is_terminal=is_terminal, position=new_position, velocity=new_velocity)

        return reward, new_state

    def initialize_policy(self, policy: Policy, policy_parameters: common.PolicyParameters):
        self._start_state_distribution.print_hello()
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
