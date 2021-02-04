from typing import TypeVar, Generic
from dataclasses import dataclass

import utils

T = TypeVar('T')


class Recorder(Generic[T]):
    def __init__(self):
        self.tallies: dict[T, _Tally] = {}

    def reset(self):
        self.tallies = {}

    def __getitem__(self, key: T) -> float:
        if key in self.tallies:
            return self.tallies[key].average
        else:
            return 0.0

    def __setitem__(self, key: T, value: float):
        if key in self.tallies:
            tally = self.tallies[key]
            tally.count += 1
            tally.average += (1.0 / tally.count) * (value - tally.average)
        else:
            self.tallies[key] = _Tally(count=1, average=value)

    @property
    def key_type(self) -> type:
        return utils.get_generic_types(self)[0]


@dataclass
class _Tally:
    count: int
    average: float
