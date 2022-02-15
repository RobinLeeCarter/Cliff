from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model.environment.non_tabular.dimension.float_dimension import FloatDimension

from mdp.common.distribution.continuous import ContinuousDistribution
from mdp.scenarios.mountain_car.model.state import State


class StartStateDistribution(ContinuousDistribution[State]):
    def __init__(self,
                 position_dimension: FloatDimension
                 ):
        self._position_dimension = position_dimension

    def draw_one(self) -> State:
        # "Samples are uniformly distributed over the half-open interval [low, high)"
        random_position = np.random.uniform(low=self._position_dimension.min,
                                            high=self._position_dimension.max)
        return State(is_terminal=False, position=random_position, velocity=0.0)
