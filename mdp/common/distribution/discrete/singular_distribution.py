from __future__ import annotations
from typing import TypeVar, Generic

from mdp.common.distribution.discrete.uniform_multinoulli import UniformMultinoulli

T = TypeVar('T')


class SingularDistribution(Generic[T], UniformMultinoulli[T]):
    def __init__(self,
                 values: list[T]
                 ):
        super().__init__(values)
        self._single_value: T = values[0]

    def draw_one(self) -> T:
        return self._single_value
