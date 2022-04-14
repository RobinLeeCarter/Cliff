from __future__ import annotations
from abc import ABC

from mdp.model.non_tabular.algorithm.abstract.nontabular_episodic import NonTabularEpisodic


class BatchEpisodic(NonTabularEpisodic, ABC):
    # start of episodes
    def start_episodes(self):
        pass

    # end of episodes
    def end_episodes(self):
        pass
