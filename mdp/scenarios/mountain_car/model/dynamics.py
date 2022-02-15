from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.scenarios.mountain_car.model.action import Action
    from mdp.scenarios.mountain_car.model.environment import Environment
    from mdp.scenarios.mountain_car.model.environment_parameters import EnvironmentParameters

from mdp import common
from mdp.model.environment.non_tabular import non_tabular_dynamics

from mdp.scenarios.mountain_car.model.state import State
from mdp.scenarios.mountain_car.model.start_state_distribution import StartStateDistribution


class Dynamics(non_tabular_dynamics.NonTabularDynamics):
    def __init__(self, environment_: Environment, environment_parameters: EnvironmentParameters):
        super().__init__(environment_, environment_parameters)

        # downcast
        self._environment: Environment = self._environment

    def build(self):
        super().build()

    def get_start_state_distribution(self) -> common.Distribution[State]:
        """
        Starting state distribution
        If want to use something different to a Uniform list of States, override this method to return the distribution
        """
        return StartStateDistribution(position_dimension=self._environment.position_dimension)

    def draw_response(self, state: State, action: Action) -> tuple[float, State]:
        """
        draw a single outcome for a single state and action
        standard call for episodic algorithms
        """
        position_dimension = self._environment.position_dimension
        velocity_dimension = self._environment.velocity_dimension
        new_position: float
        new_velocity: float
        is_terminal: bool = False
        reward: float = -1.0

        # rules from Sutton and Barto RL 10.1 p245
        new_position = position_dimension.bound(state.position + state.velocity)
        if new_position == position_dimension.min:
            new_velocity = 0.0
        elif new_position == position_dimension.max:
            new_velocity = 0.0
            is_terminal = True
            reward = 0.0
        else:
            # ẋ(t) + 0.001*A(t) - 0.0025*cos( 3 * x(t) )
            projected_velocity = state.velocity + 0.001 * action.acceleration - 0.0025 * math.cos(3.0 * state.position)
            new_velocity = velocity_dimension.bound(projected_velocity)
        new_state = State(is_terminal=is_terminal, position=new_position, velocity=new_velocity)

        return reward, new_state
