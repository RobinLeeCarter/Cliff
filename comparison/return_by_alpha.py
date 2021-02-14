from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import view
import common
from comparison import comparison_, recorder


class ReturnByAlpha(comparison_.Comparison):
    def __init__(self, scenario: common.AlgorithmByAlpha, graph: view.Graph, verbose: bool = False):
        super().__init__(scenario, graph, verbose)
        assert isinstance(self.scenario, common.AlgorithmByAlpha)
        self.scenario: common.AlgorithmByAlpha = self.scenario

        recorder_key_type = tuple[common.AlgorithmType, float]
        self._recorder = recorder.Recorder[recorder_key_type]()
        self._y_label = "Average Return"

    def record(self):
        trainer = self._trainer
        settings = trainer.settings
        algorithm_type = settings.algorithm_type
        alpha = settings.algorithm_parameters.alpha
        total_return = trainer.episode.total_return
        self._recorder[algorithm_type, alpha] = total_return

    def compile(self):
        self.x_series = common.Series(
            title="α",
            values=np.array(self.scenario.alpha_list)
        )
        # collate output from self.recorder
        for algorithm_type in self.scenario.algorithm_type_list:
            values = np.array(
                [self._recorder[algorithm_type, alpha] for alpha in self.scenario.alpha_list],
                dtype=float
            )
            title = common.algorithm_name[algorithm_type]
            series_ = common.Series(
                title=title,
                values=values,
                identifiers={"algorithm_type": algorithm_type}
            )
            self.series_list.append(series_)

    def draw_graph(self):
        gp = self.scenario.graph_parameters
        self.graph.make_plot(x_series=self.x_series,
                             graph_series=self.series_list,
                             y_label=self._y_label,
                             x_min=self.scenario.alpha_min,
                             x_max=self.scenario.alpha_max,
                             y_min=gp.y_min,
                             y_max=gp.y_max
                             )
