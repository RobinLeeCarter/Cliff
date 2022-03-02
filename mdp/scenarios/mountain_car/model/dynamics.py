from __future__ import annotations

# import math
from typing import TYPE_CHECKING

from mdp.model.environment.dynamics import Response

if TYPE_CHECKING:
    # from mdp.model.algorithm.abstract.algorithm import Algorithm
    # from mdp.model.policy.policy import Policy
    # from mdp.model.algorithm.value_function import state_function
    # from mdp.model.environment.non_tabular.non_tabular_state import NonTabularState
    # from mdp.model.environment.non_tabular.non_tabular_action import NonTabularAction
    from mdp.scenarios.mountain_car.model.environment import Environment
    from mdp.scenarios.mountain_car.model.environment_parameters import EnvironmentParameters

# from mdp import common
from mdp.model.environment.non_tabular.non_tabular_environment import NonTabularDynamics
# from mdp.model.environment.non_tabular.non_tabular_environment import NonTabularEnvironment
# from mdp.model.environment.non_tabular.dimension.float_dimension import FloatDimension
# from mdp.model.environment.non_tabular.dimension.category_dimension import CategoryDimension

from mdp.scenarios.mountain_car.model.state import State
from mdp.scenarios.mountain_car.model.action import Action
# from mdp.scenarios.mountain_car.model.start_state_distribution import StartStateDistribution
# from mdp.scenarios.mountain_car.enums import Dim


class Dynamics(NonTabularDynamics[State, Action]):
    def __init__(self, environment: Environment, environment_parameters: EnvironmentParameters):
        super().__init__(environment, environment_parameters)
        self._environment: Environment = environment
        print("Dynamics init")
        print(type(self._environment))

    def draw_response(self, state: State, action: Action) -> Response:
        self._environment.mountain()
        return 0.0, state
