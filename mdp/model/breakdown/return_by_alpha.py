from __future__ import annotations

import copy

import numpy as np

from mdp import common
from mdp.model.breakdown.recorder import Recorder
from mdp.model.breakdown.base_breakdown import BaseBreakdown


class ReturnByAlpha(BaseBreakdown):
    def __init__(self, comparison: common.Comparison):
        super().__init__(comparison)
        assert isinstance(self.comparison.breakdown_parameters, common.BreakdownAlgorithmByAlpha)
        self.breakdown_parameters: common.BreakdownAlgorithmByAlpha = self.comparison.breakdown_parameters

        # AlgorithmType, alpha
        recorder_key_type = tuple[common.AlgorithmType, float]
        self._recorder = Recorder[recorder_key_type]()
        self._y_label = "Average Return"

    def record(self):
        trainer = self._trainer
        settings = trainer.settings
        algorithm_type = settings.algorithm_parameters.algorithm_type
        alpha = settings.algorithm_parameters.alpha
        total_return = trainer.episode.total_return
        self._recorder[algorithm_type, alpha] = total_return

    def compile(self):
        self.x_series = common.Series(
            title="α",
            values=np.array(self.breakdown_parameters.alpha_list)
        )
        # collate output from self.recorder
        for algorithm_type in self.breakdown_parameters.algorithm_type_list:
            values = np.array(
                [self._recorder[algorithm_type, alpha] for alpha in self.breakdown_parameters.alpha_list],
                dtype=float
            )
            name = self._trainer.agent.algorithm_factory.get_algorithm_name(algorithm_type)
            series = common.Series(
                title=name,
                values=values,
                identifiers={"algorithm_type": algorithm_type}
            )
            self.series_list.append(series)

        # title: str = self._trainer.agent.algorithm_factory.get_algorithm_title(settings.algorithm_parameters)
        # for settings in self.comparison.settings_list:
        #     algorithm_parameters: common.AlgorithmParameters = settings.algorithm_parameters
        #     algorithm_type: common.AlgorithmType = algorithm_parameters.algorithm_type
        #     alpha: float = algorithm_parameters.alpha
        #     values = np.array(
        #         [self._recorder[algorithm_type, alpha] for alpha in self.breakdown_parameters.alpha_list],
        #         dtype=float
        #     )

    def get_graph2d_values(self) -> common.Graph2DValues:
        graph_values: common.Graph2DValues = copy.deepcopy(self.comparison.graph2d_values)
        graph_values.x_series = self.x_series
        graph_values.graph_series = self.series_list
        graph_values.y_label = self._y_label
        graph_values.x_min = self.breakdown_parameters.alpha_min
        graph_values.x_max = self.breakdown_parameters.alpha_max
        return graph_values
