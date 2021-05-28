from __future__ import annotations
from typing import TypeVar, Generic
import copy

import utils
from mdp import common

T = TypeVar('T')


class Recorder(Generic[T]):
    def __init__(self):
        self.tallies: dict[T, common.Tally] = {}

    def reset(self):
        self.tallies = {}

    @property
    def key_type(self) -> type:
        return utils.get_generic_types(self)[0]

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
            self.tallies[key] = common.Tally(count=1, average=value)

    def add_recorder(self, recorder: Recorder[T]):
        for key, tally in recorder.tallies.items():
            if key in self.tallies:
                existing_tally = self.tallies[key]
                new_count = existing_tally.count + tally.count
                new_total = existing_tally.count * existing_tally.average + tally.count * tally.average
                new_average = new_total / new_count
                existing_tally.count = new_count
                existing_tally.average = new_average
            else:
                self.tallies[key] = copy.copy(tally)
