from __future__ import annotations

import copy
from typing import Optional, TYPE_CHECKING

import numpy as np

import utils

if TYPE_CHECKING:
    from mdp.task.mountain_car.controller import Controller
    from mdp.task.mountain_car.model.environment_parameters import EnvironmentParameters
    from mdp.model.non_tabular.feature.base_feature import BaseFeature
    from mdp.model.non_tabular.environment.dimension.dims import Dims
    from mdp.model.non_tabular.environment.dimension.float_dimension import FloatDimension
    from mdp.model.non_tabular.value_function.state_action.state_action_function import StateActionFunction
from mdp import common
from mdp.model.non_tabular.non_tabular_model import NonTabularModel
# from mdp.model.non_tabular.feature.tile_coding.tile_coding import TileCoding
from mdp.task.mountain_car.model.state import State
from mdp.task.mountain_car.model.action import Action
from mdp.task.mountain_car.model.environment import Environment
from mdp.task.mountain_car.enums import Dim

# from mdp.model.non_tabular.environment.dimension.dims import Dims
# from mdp.task.mountain_car.enums import Dim


class Model(NonTabularModel[State, Action, Environment],
            environment_type=common.EnvironmentType.MOUNTAIN_CAR):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self._controller: Optional[Controller] = None
        self._feature: Optional[BaseFeature] = None

    def _create_environment(self, environment_parameters: EnvironmentParameters) -> Environment:
        return Environment(environment_parameters)

    def get_state_action_graph(self) -> common.Graph3DValues:
        g: common.Graph3DValues = copy.copy(self._comparison.graph3d_values)

        dims: Dims = self.environment.dims
        position_dim: FloatDimension = dims.state_float_dims[Dim.POSITION]
        velocity_dim: FloatDimension = dims.state_float_dims[Dim.VELOCITY]
        position_values: list[float] = utils.float_range_steps(start=position_dim.min,
                                                               stop=position_dim.max,
                                                               steps=g.steps)
        velocity_values: list[float] = utils.float_range_steps(start=velocity_dim.min,
                                                               stop=velocity_dim.max,
                                                               steps=g.steps)

        x_values = np.array(position_values, dtype=float)
        y_values = np.array(velocity_values, dtype=float)
        z_values = np.empty(shape=y_values.shape + x_values.shape, dtype=float)

        q: StateActionFunction = self.algorithm.Q
        possible_actions: list[Action] = self.environment.actions

        for x, position in enumerate(position_values):
            for y, velocity in enumerate(velocity_values):
                state: State = State(
                    is_terminal=False,
                    position=position,
                    velocity=velocity
                )
                # self.environment.build_possible_actions(state, build_array=False)
                # possible_actions: list[Action] = self.environment.possible_actions_list
                action_values: np.ndarray = q.get_action_values2(state, possible_actions)
                max_action_value: float = np.max(action_values)
                z_values[y, x] = -max_action_value

        g.x_series = common.Series(title=g.x_label, values=x_values)
        g.y_series = common.Series(title=g.y_label, values=y_values)
        g.z_series = common.Series(title=g.z_label, values=z_values)
        g.x_min = position_dim.min
        g.x_max = position_dim.max
        g.y_min = velocity_dim.min
        g.y_max = velocity_dim.max
        g.z_min = 0.0
        return g
