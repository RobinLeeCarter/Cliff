from __future__ import annotations
from abc import ABC

from mdp.model.tabular.algorithm.abstract.episodic_online import EpisodicOnline


class EpisodicOnlineControl(EpisodicOnline, ABC):
    def initialize(self):
        super().initialize()
        self._set_target_policy_greedy_wrt_q()
