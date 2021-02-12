from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import algorithm
    import view
import common
from comparison import comparison_m, recorder


class ReturnByEpisode(comparison_m.Comparison):
    def __init__(self, algorithm_factory: algorithm.Factory, graph: view.Graph, verbose: bool = False):
        super().__init__(algorithm_factory, graph, verbose)
        recorder_key_type = tuple[type, int]
        self._recorder = recorder.Recorder[recorder_key_type]()
        self._y_label = "Average Return"

    def build(self):
        self.settings_list = [
          # settings.Settings(algorithm.ExpectedSarsa, {"alpha": 0.9}),
          # settings.Settings(algorithm.VQ, {"alpha": 0.2}),
          common.Settings(common.AlgorithmType.QLearning, {"alpha": 0.5}),
          common.Settings(common.AlgorithmType.Sarsa, {"alpha": 0.5})
        ]

    def record(self):
        trainer = self._trainer
        algorithm_type = trainer.settings.algorithm_type
        episode_counter = trainer.episode_counter
        total_return = trainer.episode.total_return
        self._recorder[algorithm_type, episode_counter] = total_return

    def compile(self):
        assumed_settings = self.settings_list[0]
        start = assumed_settings.episode_to_start_recording
        frequency = assumed_settings.episode_recording_frequency
        episode_array = np.array([
            episode_counter
            for episode_counter in range(1, assumed_settings.training_episodes + 1)
            if self._is_record_episode(episode_counter, start, frequency)
        ], dtype=float)

        self.x_series = common.Series(
            title="Episode",
            values=episode_array
        )

        # collate output from self.recorder
        for settings_ in self.settings_list:
            values = np.array(
                [self._recorder[settings_.algorithm_type, episode_counter] for episode_counter in episode_array],
                dtype=float
            )
            series_ = common.Series(
                title=settings_.algorithm_title,
                identifiers={"algorithm_type": settings_.algorithm_type},
                values=values
            )
            self.series_list.append(series_)

    def draw_graph(self):
        assumed_settings = self.settings_list[0]
        self.graph.make_plot(x_series=self.x_series,
                             graph_series=self.series_list,
                             y_label=self._y_label,
                             moving_average_window_size=assumed_settings.moving_average_window_size,
                             x_min=0,
                             x_max=assumed_settings.training_episodes,
                             y_min=-100,
                             y_max=0
                             )
