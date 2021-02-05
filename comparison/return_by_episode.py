import numpy as np

import constants
import algorithm
import train
from comparison import series, comparison


class ReturnByEpisode(comparison.Comparison):
    def __init__(self, recorder: train.Recorder):
        super().__init__(recorder)

        self._training_episodes = constants.TRAINING_ITERATIONS

    def build_settings(self):
        self.settings_list = [
          algorithm.Settings(algorithm.ExpectedSarsa, {"alpha": 0.9}),
          algorithm.Settings(algorithm.VQ, {"alpha": 0.2}),
          algorithm.Settings(algorithm.QLearning, {"alpha": 0.9}),
          algorithm.Settings(algorithm.SarsaAlg, {"alpha": 0.9})
        ]

    def compile_series(self):
        self.x_series = series.Series(
            title="Iterations",
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
