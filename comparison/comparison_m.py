from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING


if TYPE_CHECKING:
    import common
    import algorithm
    import view
    from comparison import recorder
from comparison import trainer


class Comparison(ABC):
    def __init__(self, algorithm_factory: algorithm.Factory, graph: view.Graph, verbose: bool = False):
        self.algorithm_factory: algorithm.Factory = algorithm_factory
        self.graph = graph
        self.verbose = verbose

        self._trainer = trainer.Trainer(
            algorithm_factory=self.algorithm_factory,
            comparison_=self,
            verbose=False
        )
        self._recorder: Optional[recorder.Recorder] = None
        self.settings_list: list[common.Settings] = []
        self.x_series: Optional[common.Series] = None
        self.series_list: list[common.Series] = []
        self._y_label: str = ""

    @abstractmethod
    def build(self):
        pass

    def train(self, settings: common.Settings):
        self._trainer.train(settings)

    def review(self):
        episode_counter = self._trainer.episode_counter
        start = self._trainer.settings.episode_to_start_recording
        frequency = self._trainer.settings.episode_recording_frequency
        if self._is_record_episode(episode_counter, start, frequency):
            self.record()

    def _is_record_episode(self, episode_counter: int, start: int, frequency: int) -> bool:
        return episode_counter >= start and episode_counter % frequency == 0

    @abstractmethod
    def record(self):
        pass

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def draw_graph(self):
        pass
