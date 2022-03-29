from __future__ import annotations
from abc import ABC

from mdp.model.tabular.algorithm.abstract.tabular_episodic_online import TabularEpisodicOnline


class EpisodicOnlineControl(TabularEpisodicOnline, ABC):
    def initialize(self):
        super().initialize()
        self._set_target_policy_greedy_wrt_q()
