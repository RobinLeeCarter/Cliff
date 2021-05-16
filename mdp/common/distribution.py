from __future__ import annotations
from typing import TypeVar, Generic, Optional
import math
import numpy as np
from mdp.common.dict_zero import DictZero
import utils


T = TypeVar('T')


class Distribution(Generic[T]):
    def __init__(self, full_dict: Optional[dict] = None):
        self._k: list[T] = []
        self._p: np.ndarray = np.array(0, dtype=float)
        self._cum_p: np.ndarray = np.array(0, dtype=float)
        if full_dict:
            self._dict = full_dict
            self.seal()
        else:
            self._dict: DictZero = DictZero()

    def __setitem__(self, key: T, value: float):
        self._dict[key] = value

    def set_dict(self, full_dict: dict):
        self._dict = full_dict
        self.seal()

    @property
    def dict(self) -> dict:
        return self._dict

    def __getitem__(self, item: T) -> float:
        return self._dict[item]

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def keys(self):
        return self._dict.keys()

    def seal(self, do_self_check: bool = True):
        """must call this before drawing examples from distribution"""
        if do_self_check:
            self._self_check()
        self._build_arrays()

    def _self_check(self):
        cumulative_probability: float = 0.0
        for key, probability in self._dict.items():
            if 0.0 <= probability <= 1.0:
                cumulative_probability += probability
            else:
                raise ValueError(f"Probability {probability} is not between 0.0 and 1.0, for {key}")
        if not math.isclose(cumulative_probability, 1.0):
            types = set(type(key) for key in self._dict.keys())
            raise ValueError(f"Distribution sums to {cumulative_probability} instead of 1.0, "
                             f"for distribution over {types.pop()}")

    def _build_arrays(self):
        self._k = [k for k, p in self._dict.items() if p > 0.0]
        self._p = np.array([p for p in self._dict.values() if p > 0.0], dtype=float)
        self._cum_p = np.cumsum(self._p)
        # size: int = sum(1 for p in self._dict.values() if p > 0.0)
        # self._p = np.empty(size, dtype=float)
        # self._k = np.empty(size, dtype=T)
        # i: int = 0
        # for key, probability in self._dict.items():
        #     if probability > 0.0:
        #         self._k[i] = key
        #         self.

    def draw_one(self) -> T:
        return self._k[utils.cum_p_choice(self._cum_p)]

        # return random.choices(
        #     population=list(self.keys()),
        #     weights=list(self.values())
        # )[0]
