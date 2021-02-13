from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import view
import common
from comparison import comparison_m, recorder


class ReturnByEpisode(comparison_m.Comparison):
    def __init__(self, scenario: common.Scenario, graph: view.Graph, verbose: bool = False):
        super().__init__(scenario, graph, verbose)
        recorder_key_type = tuple[common.AlgorithmType, int]
        self._recorder = recorder.Recorder[recorder_key_type]()
        self._y_label = "Average Return"

    def record(self):
        trainer = self._trainer
        algorithm_type = trainer.settings.algorithm_type
        episode_counter = trainer.episode_counter
        total_return = trainer.episode.total_return
        self._recorder[algorithm_type, episode_counter] = total_return

    def compile(self):
        scenario_settings = self.scenario.scenario_settings
        start = scenario_settings.episode_to_start_recording
        frequency = scenario_settings.episode_recording_frequency
        episode_array = np.array([
            episode_counter
            for episode_counter in range(1, scenario_settings.training_episodes + 1)
            if self._is_record_episode(episode_counter, start, frequency)
        ], dtype=float)

        self.x_series = common.Series(
            title="Episode",
            values=episode_array
        )

        # collate output from self.recorder
        for settings_ in self.scenario.settings_list:
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
        scenario_settings = self.scenario.scenario_settings
        gp = self.scenario.graph_parameters
        moving_average_window_size: Optional[int] = gp.get("moving_average_window_size", None)
        y_min: Optional[float] = gp.get("y_min", None)
        y_max: Optional[float] = gp.get("y_min", None)

        self.graph.make_plot(x_series=self.x_series,
                             graph_series=self.series_list,
                             y_label=self._y_label,
                             moving_average_window_size=moving_average_window_size,
                             x_min=0,
                             x_max=scenario_settings.training_episodes,
                             y_min=y_min,
                             y_max=y_max
                             )
