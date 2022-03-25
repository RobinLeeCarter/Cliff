from __future__ import annotations
from dataclasses import dataclass

from mdp.model.non_tabular.environment.dimension.dimension import Dimension


@dataclass
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
