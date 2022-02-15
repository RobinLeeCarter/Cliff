from __future__ import annotations
import dataclasses
import math

from mdp.model.environment.non_tabular.dimension.dimension import Dimension


@dataclasses.dataclass
class FloatDimension(Dimension):
    min: float
    max: float
    wrap_around: bool = False

    @property
    def range(self) -> float:
        return self.max - self.min

    def bound(self, x: float) -> float:
        """return x bound by >= min and <= max"""
        if x <= self.min:
            return self.min
        if x >= self.max:
            return self.max
        else:
            return x

    def calc_tile_size(self, tiles: int) -> float:
        return self.range / tiles

    def calc_tiles(self, tile_size: float) -> int:
        if tile_size == 0.0:
            return 0
        else:
            return math.ceil(self.range / tile_size)
