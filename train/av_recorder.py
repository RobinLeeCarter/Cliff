from typing import Dict

from algorithm import settings
from dataclasses import dataclass


class AvRecorder:
    def __init__(self):
        self.tally: Dict[_AvKey, _Tally] = {}

    def __getitem__(self, setting_iteration_tuple: tuple[settings.Settings, int]) -> float:
        settings_, iteration = setting_iteration_tuple
        av_key = _AvKey(settings_, iteration)
        if av_key in self.tally:
            return self.tally[av_key].av_return
        else:
            return 0.0

    def __setitem__(self, setting_iteration_tuple: tuple[settings.Settings, int], value: float):
        settings_, iteration = setting_iteration_tuple
        av_key = _AvKey(settings_, iteration)
        if av_key in self.tally:
            tally = self.tally[av_key]
            tally.count += 1
            tally.av_return = (1.0/tally.count)*(value - tally.av_return)


@dataclass(frozen=True)
class _AvKey:
    settings: settings.Settings
    iteration: int


@dataclass
class _Tally:
    count: int
    av_return: float
