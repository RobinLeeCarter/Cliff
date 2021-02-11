from __future__ import annotations

import common
import comparison


class ReturnByEpisode(comparison.ReturnByEpisode):
    def build(self):
        self.settings_list = [
          common.Settings(common.AlgorithmType.Sarsa, {"alpha": 0.5})
        ]

    def draw_graph(self):
        assumed_settings = self.settings_list[0]
        self.graph.make_plot(x_series=self.x_series,
                             graph_series=self.series_list,
                             moving_average_window_size=assumed_settings.moving_average_window_size,
                             x_min=0,
                             x_max=assumed_settings.training_episodes,
                             y_min=-100,
                             y_max=0
                             )
