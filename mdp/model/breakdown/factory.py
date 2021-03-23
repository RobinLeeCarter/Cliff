from __future__ import annotations
from typing import Optional

from mdp import common
from mdp.model.breakdown import return_by_alpha, breakdown_, return_by_episode, rms_by_episode, episode_by_timestep


def factory(comparison: common.Comparison) -> Optional[breakdown_.Breakdown]:
    c = common.BreakdownType
    breakdown_type = comparison.breakdown_parameters.breakdown_type
    breakdown: Optional[breakdown_.Breakdown]
    if breakdown_type == c.EPISODE_BY_TIMESTEP:
        breakdown = episode_by_timestep.EpisodeByTimestep(comparison)
    elif breakdown_type == c.RETURN_BY_EPISODE:
        breakdown = return_by_episode.ReturnByEpisode(comparison)
    elif breakdown_type == c.RMS_BY_EPISODE:
        breakdown = rms_by_episode.RmsByEpisode(comparison)
    elif breakdown_type == c.RETURN_BY_ALPHA:
        breakdown = return_by_alpha.ReturnByAlpha(comparison)
    else:
        breakdown = None

    return breakdown
