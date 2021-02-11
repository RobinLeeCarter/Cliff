from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import algorithm
    import view
import common
from comparison import comparison_m, recorder


class EpisodeByTimestep(comparison_m.Comparison):
    def __init__(self, algorithm_factory: algorithm.Factory, graph: view.Graph, verbose: bool = False):
        super().__init__(algorithm_factory, graph, verbose)
        recorder_key_type = tuple[int, int]
        self._recorder = recorder.Recorder[recorder_key_type]()
        self._max_timestep: int = 0

    def build(self):
        self.settings_list = [
          # settings.Settings(algorithm.ExpectedSarsa, {"alpha": 0.9}),
          # settings.Settings(algorithm.VQ, {"alpha": 0.2}),
          # common.Settings(common.AlgorithmType.QLearning, {"alpha": 0.5}),
          common.Settings(common.AlgorithmType.Sarsa, {"alpha": 0.5})
        ]
        self._trainer.review_every_step = True

    def record(self):
        trainer = self._trainer
        algorithm_type = trainer.settings.algorithm_type
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
        for settings_ in self.settings_list:
            values = np.array(
                [self._recorder[settings_.algorithm_type, timestep] for timestep in timestep_array],
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
                             x_min=0,
                             x_max=self._max_timestep,
                             y_min=0,
                             y_max=assumed_settings.training_episodes
                             )
