from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from comparison import recorder, comparison_dataclasses
    import agent
    import view


class Comparison(ABC):
    def __init__(self, graph: view.Graph):
        self._recorder: Optional[recorder.Recorder] = None
        self.settings_list: list[comparison_dataclasses.Settings] = []
        self.x_series: Optional[comparison_dataclasses.Series] = None
        self.series_list: list[comparison_dataclasses.Series] = []
        self.graph = graph

    @abstractmethod
    def build(self):
        pass

    def review(self, settings_: comparison_dataclasses.Settings, iteration: int, episode: agent.Episode):
        if self._is_record_iteration(settings_, iteration):
            self.record(settings_, iteration, episode)

    def _is_record_iteration(self, settings_: comparison_dataclasses.Settings, iteration: int) -> bool:
        return iteration >= settings_.performance_sample_start and \
               iteration % settings_.performance_sample_frequency == 0

    @abstractmethod
    def record(self, settings_: comparison_dataclasses.Settings, iteration: int, episode: agent.Episode):
        pass

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def draw_graph(self):
        pass
