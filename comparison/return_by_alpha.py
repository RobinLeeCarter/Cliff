from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import view
    import algorithm
import utils
import common
from comparison import comparison_m, recorder


class ReturnByAlpha(comparison_m.Comparison):
    def __init__(self, algorithm_factory: algorithm.Factory, graph: view.Graph, verbose: bool = False):
        super().__init__(algorithm_factory, graph, verbose)

        self._algorithm_type_list = [
            common.AlgorithmType.ExpectedSarsa,
            common.AlgorithmType.VQ,
            common.AlgorithmType.QLearning,
            common.AlgorithmType.Sarsa
        ]
        self._alpha_min = 0.1
        self._alpha_max = 1.0
        self._alpha_step = 0.1
        self._alpha_list = utils.float_range(start=self._alpha_min, stop=self._alpha_max, step_size=self._alpha_step)
        recorder_key_type = tuple[type, float]
        self._recorder = recorder.Recorder[recorder_key_type]()
        self._y_label = "Average Return"

    def build(self):
        self.settings_list = []
        for alpha in self._alpha_list:
            for algorithm_type in self._algorithm_type_list:
                settings_ = common.Settings(
                    algorithm_type=algorithm_type,
                    parameters={"alpha": alpha}
                )
                self.settings_list.append(settings_)

    def record(self):
        trainer = self._trainer
        settings = trainer.settings
        algorithm_type = settings.algorithm_type
        alpha = settings.algorithm_parameters["alpha"]
        total_return = trainer.episode.total_return
        self._recorder[algorithm_type, alpha] = total_return

    def compile(self):
        self.x_series = common.Series(
            title="α",
            values=np.array(self._alpha_list)
        )
        # collate output from self.recorder
        for algorithm_type in self._algorithm_type_list:
            values = np.array(
                [self._recorder[algorithm_type, alpha] for alpha in self._alpha_list],
                dtype=float
            )
            series_ = common.Series(
                title=algorithm_type.name,
                values=values,
                identifiers={"algorithm_type": algorithm_type}
            )
            self.series_list.append(series_)

    def draw_graph(self):
        # assumed_settings = self.settings_list[0]
        self.graph.make_plot(x_series=self.x_series,
                             graph_series=self.series_list,
                             y_label=self._y_label,
                             x_min=self._alpha_min,
                             x_max=self._alpha_max,
                             y_min=-140,
                             y_max=0
                             )
