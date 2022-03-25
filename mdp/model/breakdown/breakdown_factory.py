from __future__ import annotations
from typing import Type, Optional

from mdp import common
from mdp.model.breakdown.base_breakdown import BaseBreakdown
from mdp.model.breakdown.return_by_alpha import ReturnByAlpha
from mdp.model.breakdown.rms_by_episode import RmsByEpisode
from mdp.model.breakdown.return_by_episode import ReturnByEpisode
from mdp.model.breakdown.episode_by_timestep import EpisodeByTimestep


class BreakdownFactory:
    def __init__(self):
        bt = common.BreakdownType
        self._breakdown_lookup: dict[bt, Type[BaseBreakdown]] = {
            bt.EPISODE_BY_TIMESTEP: EpisodeByTimestep,
            bt.RETURN_BY_EPISODE: ReturnByEpisode,
            bt.RMS_BY_EPISODE: RmsByEpisode,
            bt.RETURN_BY_ALPHA: ReturnByAlpha
        }

    def create(self, comparison: common.Comparison) -> Optional[BaseBreakdown]:
        breakdown_type: common.BreakdownType = comparison.breakdown_parameters.breakdown_type
        if breakdown_type:
            type_of_breakdown: Type[BaseBreakdown] = self._breakdown_lookup[breakdown_type]
            breakdown: BaseBreakdown = type_of_breakdown(comparison)
            return breakdown
        else:
            return None
