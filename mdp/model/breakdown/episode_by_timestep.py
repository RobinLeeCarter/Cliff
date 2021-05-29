from __future__ import annotations

import numpy as np

import utils
from mdp import common
from mdp.model.breakdown import recorder, breakdown


class EpisodeByTimestep(breakdown.Breakdown):
    def __init__(self, comparison: common.Comparison):
        super().__init__(comparison)

        # algorithm_type, timestep
        recorder_key_type = tuple[common.AlgorithmType, int]
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
        for settings in self.comparison.settings_list:
            values = np.array(
                [self._recorder[settings.algorithm_parameters.algorithm_type, timestep]
                 for timestep in timestep_array],
                dtype=float
            )
            series_ = common.Series(
                title=settings.algorithm_title,
                identifiers={"algorithm_type": settings.algorithm_parameters.algorithm_type},
                values=values
            )
            self.series_list.append(series_)

    def get_graph_values(self) -> common.GraphValues:
        graph_values: common.GraphValues = common.GraphValues(
            x_series=self.x_series,
            graph_series=self.series_list,
            y_label=self._y_label,
            x_min=0,
            x_max=self._max_timestep,
            y_min=0,
            y_max=self.comparison.comparison_settings.training_episodes
        )
        utils.set_none_to_default(graph_values, self.comparison.graph_values)
        return graph_values
