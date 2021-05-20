from __future__ import annotations
import abc

from mdp.model.algorithm.abstract.episodic_online import EpisodicOnline


class EpisodicOnlineControl(EpisodicOnline, abc.ABC):
    def initialize(self):
        super().initialize()
        self._set_target_policy_greedy_wrt_q()
