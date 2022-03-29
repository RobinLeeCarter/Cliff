from __future__ import annotations

from typing import Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from mdp.task.mountain_car.controller import Controller
    from mdp.task.mountain_car.model.environment_parameters import EnvironmentParameters
    from mdp.model.non_tabular.feature.feature import Feature
    # from mdp import common
from mdp.model.non_tabular.non_tabular_model import NonTabularModel
# from mdp.model.non_tabular.feature.tile_coding.tile_coding import TileCoding
from mdp.task.mountain_car.model.state import State
from mdp.task.mountain_car.model.action import Action
from mdp.task.mountain_car.model.environment import Environment

# from mdp.model.non_tabular.environment.dimension.dims import Dims
# from mdp.task.mountain_car.enums import Dim


class Model(NonTabularModel[State, Action, Environment]):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self._controller: Optional[Controller] = None
        self._feature: Optional[Feature] = None

    def _create_environment(self, environment_parameters: EnvironmentParameters) -> Environment:
        return Environment(environment_parameters)

    # def build(self, comparison: common.Comparison):
    #     super().build(comparison)
    #     # self._create_tile_coding()
    #     # create tile_coding here

    # def _create_tile_coding(self):
    #     dims: Dims = self.environment.dims
    #     self._feature: TileCoding[State, Action] = TileCoding[State, Action](dims=dims)
    #     self._feature.add(included_dims={Dim.POSITION, Dim.VELOCITY})  # , tilings=2 ** 4
    #
    #     print(f"total tilings = {self._feature.tilings}")
    #     print(f"max_size = {self._feature.max_size}")
    #
    #     tilings = self._feature.tilings
    #     alpha = 0.1 / tilings

