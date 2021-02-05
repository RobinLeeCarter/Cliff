import numpy as np

import utils
import algorithm
import agent
import train
from comparison import settings, series, comparison


class ReturnByAlpha(comparison.Comparison):
    def __init__(self):
        super().__init__()

        self._algorithm_type_list = [
            algorithm.ExpectedSarsa,
            algorithm.VQ,
            algorithm.QLearning,
            algorithm.SarsaAlg
        ]
        self._alpha_list = utils.float_range(start=0.1, stop=1.0, step_size=0.05)
        recorder_key_type = tuple[type, float]
        self.recorder = train.Recorder[recorder_key_type]()

    def build(self):
        self.settings_list = []
        for alpha in self._alpha_list:
            for algorithm_type in self._algorithm_type_list:
                settings_ = settings.Settings(
                    algorithm_type=algorithm_type,
                    parameters={"alpha": alpha}
                )
                self.settings_list.append(settings_)

    def record(self, settings_: settings.Settings, iteration: int, episode: agent.Episode):
        algorithm_type = settings_.algorithm_type
        alpha = settings_.parameters["alpha"]
        total_return = episode.total_return
        self._recorder[algorithm_type, alpha] = total_return

    def compile(self):
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
