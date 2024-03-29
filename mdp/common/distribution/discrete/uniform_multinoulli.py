from __future__ import annotations
from typing import TypeVar, Generic
import utils

from mdp.common.distribution.discrete.discrete_distribution import DiscreteDistribution

T = TypeVar('T')


class UniformMultinoulli(Generic[T], DiscreteDistribution[T]):  # , AbstractDistribution[T]
    """Uniform Discrete Distribution, different from Multinoulli for speed"""
    def __init__(self,
                 values: list[T]
                 ):
        self._values: list[T] = values
        self._n: int = len(self._values)
        self._single_value: T
        if self._n == 1:
            self._single_value = values[0]

    def draw_one(self) -> T:
        return self._values[utils.n_choice(self._n)]


# if __name__ == "__main__":
#     d = UniformDistribution[int]([12, 13, 14, 15, 16])
#     for _ in range(10):
#         print(d.draw_one())
