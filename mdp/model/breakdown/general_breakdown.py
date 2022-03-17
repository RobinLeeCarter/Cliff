from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from mdp import common
    from mdp.model.breakdown.recorder import Recorder
    from mdp.model.trainer.trainer import Trainer


class GeneralBreakdown(ABC):
    def __init__(self, comparison: common.Comparison):
        self.comparison: common.Comparison = comparison
        self.verbose: bool = self.comparison.breakdown_parameters.verbose

        self._trainer: Optional[Trainer] = None
        self._recorder: Optional[Recorder] = None
        # self.settings_list: list[common.Settings] = []
        self.x_series: Optional[common.Series] = None
        self.series_list: list[common.Series] = []
        self._y_label: str = ""

    @property
    def recorder(self) -> Optional[Recorder]:
        return self._recorder

    def set_trainer(self, trainer: Trainer):
        self._trainer = trainer

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
    def get_graph2d_values(self) -> common.Graph2DValues:
        pass
