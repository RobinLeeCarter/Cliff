from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model.environment.non_tabular.dimension.dims import Dims

from mdp.common.distribution.continuous import ContinuousDistribution
from mdp.scenarios.mountain_car.enums import Dim
from mdp.scenarios.mountain_car.model.state import State


class StartStateDistribution(ContinuousDistribution[State]):
    def __init__(self, dims: Dims):
        self._position_min = dims.state_float[Dim.POSITION].min
        self._position_max = dims.state_float[Dim.POSITION].max

    def draw_one(self) -> State:
        # "Samples are uniformly distributed over the half-open interval [low, high)"
        random_position = np.random.uniform(low=self._position_min, high=self._position_max)
        return State(is_terminal=False, position=random_position, velocity=0.0)

    def print_hello(self):
        print("hello")
