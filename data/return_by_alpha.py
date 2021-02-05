import numpy as np

import utils
import algorithm
from data import series


class ReturnByAlpha:
    def __init__(self):
        self._algorithm_type_list = [
            algorithm.ExpectedSarsa,
            algorithm.VQ,
            algorithm.QLearning,
            algorithm.SarsaAlg
        ]
        self._alpha_list = utils.float_range(start=0.1, stop=1.0, step_size=0.05)

        # for training
        self.settings = self.settings_list()

        # for output
        self.x_values = np.array(self._alpha_list)
        self.series = self.series_list()

    def settings_list(self) -> list[algorithm.Settings]:
        settings_list: list[algorithm.Settings] = []
        for alpha in self._alpha_list:
            for algorithm_type in self._algorithm_type_list:
                settings = algorithm.Settings(
                    algorithm_type=algorithm_type,
                    parameters={"alpha": alpha}
                )
                settings_list.append(settings)
        return settings_list

    def series_list(self) -> list[series.Series]:
        series_list: list[series.Series] = []
        for algorithm_type in self._algorithm_type_list:
            series_ = series.Series(
                title=algorithm_type.name,
                values=np.array([], dtype=float),
                identifiers={"algorithm_type": algorithm_type}
            )
            series_list.append(series_)
        return series_list
