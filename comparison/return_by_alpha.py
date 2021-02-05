import numpy as np

import utils
import algorithm
import train
from comparison import series, comparison


class ReturnByAlpha(comparison.Comparison):
    def __init__(self, recorder: train.Recorder):
        super().__init__(recorder)

        self._algorithm_type_list = [
            algorithm.ExpectedSarsa,
            algorithm.VQ,
            algorithm.QLearning,
            algorithm.SarsaAlg
        ]
        self._alpha_list = utils.float_range(start=0.1, stop=1.0, step_size=0.05)

    def build_settings(self):
        self.settings_list = []
        for alpha in self._alpha_list:
            for algorithm_type in self._algorithm_type_list:
                settings = algorithm.Settings(
                    algorithm_type=algorithm_type,
                    parameters={"alpha": alpha}
                )
                self.settings_list.append(settings)

    def compile_series(self):
        self.x_series = series.Series(
            title="α",
            values=np.array(self._alpha_list)
        )
        # collate output from self.recorder
        for algorithm_type in self._algorithm_type_list:
            values = np.array([self._recorder[algorithm_type, alpha] for alpha in self._alpha_list])
            series_ = series.Series(
                title=algorithm_type.name,
                values=values,
                identifiers={"algorithm_type": algorithm_type}
            )
            self.series_list.append(series_)
