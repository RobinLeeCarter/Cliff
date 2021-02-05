from abc import ABC, abstractmethod
from typing import Optional

import agent
import train
# import common
from comparison import settings, series


class Comparison(ABC):
    def __init__(self):
        self._recorder: Optional[train.Recorder] = None
        self.settings_list: list[settings.Settings] = []
        self.x_series: Optional[series.Series] = None
        self.series_list: list[series.Series] = []

    # @abstractmethod
    # def _build(self, comparison_type: common.ComparisonType):
    #     pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def record(self, settings_: settings.Settings, iteration: int, episode: agent.Episode):
        pass

    @abstractmethod
    def compile(self):
        pass
