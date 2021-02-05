from abc import ABC, abstractmethod
from typing import Optional

import agent
import view
from comparison import settings, series, recorder


class Comparison(ABC):
    def __init__(self):
        self._recorder: Optional[recorder.Recorder] = None
        self.settings_list: list[settings.Settings] = []
        self.x_series: Optional[series.Series] = None
        self.series_list: list[series.Series] = []
        self.graph = view.Graph()

    @abstractmethod
    def build(self):
        pass

    def review(self, settings_: settings.Settings, iteration: int, episode: agent.Episode):
        if self._is_record_iteration(settings_, iteration):
            self.record(settings_, iteration, episode)

    def _is_record_iteration(self, settings_: settings.Settings, iteration: int) -> bool:
        return iteration >= settings_.performance_sample_start and \
               iteration % settings_.performance_sample_frequency == 0

    @abstractmethod
    def record(self, settings_: settings.Settings, iteration: int, episode: agent.Episode):
        pass

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def draw_graph(self):
        pass
