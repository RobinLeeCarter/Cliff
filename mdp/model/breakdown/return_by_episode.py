from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import view
import common
from mdp.model.breakdown import recorder, breakdown_


class ReturnByEpisode(breakdown_.Breakdown):
    def __init__(self, comparison: common.Comparison, graph: view.Graph):
        super().__init__(comparison, graph)
        recorder_key_type = tuple[common.AlgorithmType, int]
        self._recorder = recorder.Recorder[recorder_key_type]()
        self._y_label = "Average Return"

    def record(self):
        trainer = self._trainer
        algorithm_type = trainer.settings.algorithm_parameters.algorithm_type
        episode_counter = trainer.episode_counter
        total_return = trainer.episode.total_return
        self._recorder[algorithm_type, episode_counter] = total_return

    def compile(self):
        comparison_settings = self.comparison.comparison_settings
        start = comparison_settings.episode_to_start_recording
        frequency = comparison_settings.episode_recording_frequency
        episode_array = np.array([
            episode_counter
            for episode_counter in range(1, comparison_settings.training_episodes + 1)
            if self._is_record_episode(episode_counter, start, frequency)
        ], dtype=float)

        self.x_series = common.Series(
            title="Episode",
            values=episode_array
        )

        # collate output from self.recorder
        for settings_ in self.comparison.settings_list:
            values = np.array(
                [self._recorder[settings_.algorithm_parameters.algorithm_type, episode_counter]
                 for episode_counter in episode_array],
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
        gp = self.comparison.graph_parameters
        self.graph.make_plot(x_series=self.x_series,
                             graph_series=self.series_list,
                             y_label=self._y_label,
                             moving_average_window_size=gp.moving_average_window_size,
                             x_min=0,
                             x_max=comparison_settings.training_episodes,
                             y_min=gp.y_min,
                             y_max=gp.y_max
                             )
