from __future__ import annotations
from typing import Optional, Type

from mdp import common
from mdp.model.breakdown.general_breakdown import GeneralBreakdown
from mdp.model.breakdown.return_by_alpha import ReturnByAlpha
from mdp.model.breakdown.rms_by_episode import RmsByEpisode
from mdp.model.breakdown.return_by_episode import ReturnByEpisode
from mdp.model.breakdown.episode_by_timestep import EpisodeByTimestep


class BreakdownFactory:
    def __init__(self):
        bt = common.BreakdownType
        self._breakdown_lookup: dict[bt, Type[GeneralBreakdown]] = {
            bt.EPISODE_BY_TIMESTEP: EpisodeByTimestep,
            bt.RETURN_BY_EPISODE: ReturnByEpisode,
            bt.RMS_BY_EPISODE: RmsByEpisode,
            bt.RETURN_BY_ALPHA: ReturnByAlpha
        }

    def create(self, comparison: common.Comparison):
        breakdown_type: common.BreakdownType = comparison.breakdown_parameters.breakdown_type
        type_of_breakdown: Type[GeneralBreakdown] = self._breakdown_lookup[breakdown_type]
        breakdown: GeneralBreakdown = type_of_breakdown(comparison)
        return breakdown
