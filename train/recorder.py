from typing import Dict

import algorithm
from dataclasses import dataclass


class Recorder:
    def __init__(self):
        self.tallies: Dict[_AvKey, _Tally] = {}

    def __getitem__(self, algorithm_iteration_tuple: tuple[algorithm.EpisodicAlgorithm, int]) -> float:
        algorithm_, iteration = algorithm_iteration_tuple
        av_key = _AvKey(algorithm_, iteration)
        if av_key in self.tallies:
            return self.tallies[av_key].av_return
        else:
            return 0.0

    def __setitem__(self, algorithm_iteration_tuple: tuple[algorithm.EpisodicAlgorithm, int], value: float):
        algorithm_, iteration = algorithm_iteration_tuple
        av_key = _AvKey(algorithm_, iteration)
        if av_key in self.tallies:
            tally = self.tallies[av_key]
            tally.count += 1
            tally.av_return += (1.0/tally.count)*(value - tally.av_return)
        else:
            tally = _Tally(count=1, av_return=value)
            self.tallies[av_key] = tally


@dataclass(frozen=True)
class _AvKey:
    algorithm: algorithm.EpisodicAlgorithm
    iteration: int


@dataclass
class _Tally:
    count: int
    av_return: float
