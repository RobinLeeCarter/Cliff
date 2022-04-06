from __future__ import annotations

import numpy as np
import random
from utils import numba_random


class Rng:
    """
    Storing and seeding random number generators for process-safe random processes and reproducible random processes
    """
    _rng: np.random.Generator
    max_val: int = np.iinfo(np.int32).max

    @classmethod
    def get(cls) -> np.random.Generator:
        return cls._rng

    @classmethod
    def set_by_entropy(cls):
        cls._rng = np.random.default_rng()
        cls._set_others()

    @classmethod
    def set_seed(cls, seed: int):
        cls._rng = np.random.default_rng(seed)
        cls._set_others()

    @classmethod
    def _set_others(cls):
        seeds = cls._rng.integers(low=0, high=cls.max_val, size=4)
        # np.random.seed(os.getpid() * (time.time_ns() % 12345) % 1234556789)
        np.random.seed(seeds[0])
        random.seed(seeds[1])
        # noinspection PyProtectedMember
        numba_random._set_seeds(seeds[2], seeds[3])

    @classmethod
    def get_seeds(cls, number_of_seeds: int) -> list[int]:
        # noinspection PyTypeChecker
        return cls._rng.integers(low=0, high=cls.max_val, size=number_of_seeds).tolist()


Rng.set_by_entropy()
