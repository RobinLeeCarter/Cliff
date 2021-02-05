from abc import ABC, abstractmethod
from typing import Optional

import algorithm
import train
from comparison import series


class Comparison(ABC):
    def __init__(self, recorder: train.Recorder):
        self._recorder: train.Recorder = recorder

        self.settings_list: list[algorithm.Settings] = []
        self.x_series: Optional[series.Series] = None
        self.series_list: list[series.Series] = []

    @abstractmethod
    def build_settings(self) -> list[algorithm.Settings]:
        pass

    @abstractmethod
    def compile_series(self) -> list[series.Series]:
        pass
