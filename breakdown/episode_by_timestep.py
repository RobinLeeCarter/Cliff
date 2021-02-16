from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import view
import common
from breakdown import breakdown_, recorder


class EpisodeByTimestep(breakdown_.Breakdown):
    def __init__(self, comparison: common.Comparison, graph: view.Graph):
        super().__init__(comparison, graph)
        recorder_key_type = tuple[int, int]
        self._recorder = recorder.Recorder[recorder_key_type]()
        self._max_timestep: int = 0
        self._y_label: str = "Episode"

    def record(self):
        trainer = self._trainer
        algorithm_type = trainer.settings.algorithm_parameters.algorithm_type
        timestep = trainer.timestep
        episode_counter = trainer.episode_counter
        self._recorder[algorithm_type, timestep] = episode_counter

    def compile(self):
        self._max_timestep = self._trainer.max_timestep
        timestep_array = np.arange(self._max_timestep+1, dtype=int)

        self.x_series = common.Series(
            title="Timestep",
            values=timestep_array
        )

        # collate output from self.recorder
        for settings_ in self.comparison.settings_list:
            values = np.array(
                [self._recorder[settings_.algorithm_parameters.algorithm_type, timestep]
                 for timestep in timestep_array],
                dtype=float
            )
            series_ = common.Series(
                title=settings_.algorithm_title,
                identifiers={"algorithm_type": settings_.algorithm_parameters.algorithm_type},
                values=values
            )
            self.series_list.append(series_)

    def draw_graph(self):
        comparison_settings = self.comparison.comparison_settings

        self.graph.make_plot(x_series=self.x_series,
                             graph_series=self.series_list,
                             y_label=self._y_label,
                             x_min=0,
                             x_max=self._max_timestep,
                             y_min=0,
                             y_max=comparison_settings.training_episodes
                             )
