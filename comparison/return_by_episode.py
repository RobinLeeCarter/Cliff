from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import agent
    import algorithm
    import view
import common
from comparison import comparison_m, recorder


class ReturnByEpisode(comparison_m.Comparison):
    def __init__(self, algorithm_factory: algorithm.Factory, graph: view.Graph, verbose: bool = False):
        super().__init__(algorithm_factory, graph, verbose)
        recorder_key_type = tuple[type, int]
        self._recorder = recorder.Recorder[recorder_key_type]()

    def build(self):
        self.settings_list = [
          # settings.Settings(algorithm.ExpectedSarsa, {"alpha": 0.9}),
          # settings.Settings(algorithm.VQ, {"alpha": 0.2}),
          common.Settings(common.AlgorithmType.QLearning, {"alpha": 0.5}),
          common.Settings(common.AlgorithmType.Sarsa, {"alpha": 0.5})
        ]

    def record(self, settings_: common.Settings, iteration: int, episode: agent.Episode):
        algorithm_type = settings_.algorithm_type
        total_return = episode.total_return
        self._recorder[algorithm_type, iteration] = total_return

    def compile(self):
        assumed_settings = self.settings_list[0]
        iteration_array = np.array([
            iteration
            for iteration in range(1, assumed_settings.training_iterations+1)
            if self._is_record_iteration(assumed_settings, iteration)
        ], dtype=float)

        self.x_series = common.Series(
            title="Episode",
            values=iteration_array
        )

        # collate output from self.recorder
        for settings_ in self.settings_list:
            values = np.array([
                self._recorder[settings_.algorithm_type, iteration]
                for iteration in iteration_array
            ])
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
                             moving_average_window_size=assumed_settings.moving_average_window_size,
                             x_min=0,
                             x_max=assumed_settings.training_iterations,
                             y_min=-100,
                             y_max=0
                             )
