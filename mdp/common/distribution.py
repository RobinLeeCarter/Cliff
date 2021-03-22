from __future__ import annotations
from typing import TypeVar
import math


T = TypeVar('T')


class Distribution(dict[T, float]):
    def __missing__(self, key: T) -> float:
        return 0.0

    # adding the test here halves the speed of adding an item so not included.
    # Run self_test instead if a problem.
    # def __setitem__(self, key: T, probability: float):
    #     if 0.0 <= probability <= 1.0:
    #         super().__setitem__(key, probability)
    #     else:
    #         raise ValueError(f"Probability {probability} is not between 0.0 and 1.0, for {key}")

    def self_test(self):
        cumulative_probability: float = 0.0
        for key, probability in self.items():
            if 0.0 <= probability <= 1.0:
                cumulative_probability += probability
            else:
                raise ValueError(f"Probability {probability} is not between 0.0 and 1.0, for {key}")
        if not math.isclose(cumulative_probability, 1.0):
            types = set(type(key) for key in self.keys())
            raise ValueError(f"Distribution sums to {cumulative_probability} instead of 1.0, "
                             f"for distribution over {types.pop()}")
