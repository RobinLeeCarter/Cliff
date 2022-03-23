from __future__ import annotations
import copy

import numpy as np

from mdp import common
from mdp.model.breakdown.recorder import Recorder
from mdp.model.breakdown.base_breakdown import BaseBreakdown


class ReturnByEpisode(BaseBreakdown):
    def __init__(self, comparison: common.Comparison):
        super().__init__(comparison)

        # common.AlgorithmParameters, episode
        recorder_key_type = tuple[common.AlgorithmParameters, int]
        self._recorder = Recorder[recorder_key_type]()
        self._y_label = "Average Return"

    def record(self):
        trainer = self._trainer
        algorithm_parameters = trainer.settings.algorithm_parameters
        episode_counter = trainer.episode_counter
        total_return = trainer.episode.total_return
        self._recorder[algorithm_parameters, episode_counter] = total_return

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
        for settings in self.comparison.settings_list:
            values = np.array(
                [self._recorder[settings.algorithm_parameters, episode_counter]
                 for episode_counter in episode_array],
                dtype=float
            )
            algorithm_type: common.AlgorithmType = settings.algorithm_parameters.algorithm_type
            title: str = self._trainer.algorithm_factory.get_algorithm_title(settings.algorithm_parameters)
            series = common.Series(
                title=title,
                identifiers={"algorithm_type": algorithm_type},
                values=values
            )
            self.series_list.append(series)

    def get_graph2d_values(self) -> common.Graph2DValues:
        graph_values: common.Graph2DValues = copy.deepcopy(self.comparison.graph2d_values)
        graph_values.x_series = self.x_series
        graph_values.graph_series = self.series_list
        graph_values.y_label = self._y_label
        graph_values.x_min = 0
        graph_values.x_max = self.comparison.comparison_settings.training_episodes
        return graph_values
