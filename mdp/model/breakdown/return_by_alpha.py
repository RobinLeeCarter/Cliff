from __future__ import annotations

import numpy as np

import common
from mdp.model.breakdown import recorder, breakdown_


class ReturnByAlpha(breakdown_.Breakdown):
    def __init__(self, comparison: common.Comparison):
        super().__init__(comparison)
        assert isinstance(self.comparison.breakdown_parameters, common.BreakdownAlgorithmByAlpha)
        self.breakdown_parameters: common.BreakdownAlgorithmByAlpha = self.comparison.breakdown_parameters

        recorder_key_type = tuple[common.AlgorithmType, float]
        self._recorder = recorder.Recorder[recorder_key_type]()
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
            title = common.algorithm_name[algorithm_type]
            series_ = common.Series(
                title=title,
                values=values,
                identifiers={"algorithm_type": algorithm_type}
            )
            self.series_list.append(series_)

    def get_graph_values(self) -> common.GraphValues:
        graph_values: common.GraphValues = common.GraphValues(
            x_series=self.x_series,
            graph_series=self.series_list,
            y_label=self._y_label,
            x_min=self.breakdown_parameters.alpha_min,
            x_max=self.breakdown_parameters.alpha_max,
        )
        graph_values.apply_default_to_nones(self.comparison.graph_values)
        return graph_values
