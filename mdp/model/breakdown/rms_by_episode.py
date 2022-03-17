from __future__ import annotations

from mdp import common
from mdp.model.breakdown.return_by_episode import ReturnByEpisode


class RmsByEpisode(ReturnByEpisode):
    def __init__(self, comparison: common.Comparison):
        super().__init__(comparison)
        self._y_label = "Average RMS error"

    def record(self):
        trainer = self._trainer
        algorithm_parameters = trainer.settings.algorithm_parameters
        episode_counter = trainer.episode_counter
        self._recorder[algorithm_parameters, episode_counter] = trainer.agent.rms_error()
