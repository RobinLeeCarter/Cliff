from __future__ import annotations

import numpy as np

import utils
from mdp import common
from mdp.model.breakdown.recorder import Recorder
from mdp.model.breakdown.breakdown import Breakdown


class EpisodeByTimestep(Breakdown):
    def __init__(self, comparison: common.Comparison):
        super().__init__(comparison)

        # algorithm_type, timestep
        self._recorder: Recorder[tuple[common.AlgorithmType, int]] = Recorder[tuple[common.AlgorithmType, int]]()
        self._max_timestep: int = 0
        self._y_label: str = "Episode"

    def record(self):
        trainer = self._trainer
        algorithm_type = trainer.settings.algorithm_parameters.algorithm_type
        timestep = trainer.cum_timestep
        episode_counter = trainer.episode_counter
        self._recorder[algorithm_type, timestep] = episode_counter

    def compile(self):
        # return max_timestep or get from elsewhere
        if self._trainer.max_cum_timestep > 0:
            # serial case
            self._max_timestep = self._trainer.max_cum_timestep
        else:
            # parallel case max_timestep not returned so deduce it from recoder
            self._max_timestep = max(t[1] for t in self._recorder.tallies.keys())
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
            algorithm_type: common.AlgorithmType = settings.algorithm_parameters.algorithm_type
            title: str = self._trainer.agent.algorithm_factory.lookup_algorithm_name(algorithm_type)
            series_ = common.Series(
                title=title,
                identifiers={"algorithm_type": algorithm_type},
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
