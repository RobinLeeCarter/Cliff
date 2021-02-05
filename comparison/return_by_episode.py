import numpy as np

import constants
import agent
import algorithm
import train
from comparison import settings, series, comparison


class ReturnByEpisode(comparison.Comparison):
    def __init__(self):
        super().__init__()
        recorder_key_type = tuple[type, float]
        self.recorder = train.Recorder[recorder_key_type]()

        self._training_episodes = constants.TRAINING_ITERATIONS

    def build(self):
        self.settings_list = [
          settings.Settings(algorithm.ExpectedSarsa, {"alpha": 0.9}),
          settings.Settings(algorithm.VQ, {"alpha": 0.2}),
          settings.Settings(algorithm.QLearning, {"alpha": 0.9}),
          settings.Settings(algorithm.SarsaAlg, {"alpha": 0.9})
        ]

    def record(self, settings_: settings.Settings, iteration: int, episode: agent.Episode):
        algorithm_type = settings_.algorithm_type
        total_return = episode.total_return
        self._recorder[algorithm_type, iteration] = total_return

    def compile(self):
        raise NotImplementedError
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
