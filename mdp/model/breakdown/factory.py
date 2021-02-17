from __future__ import annotations

import common
from mdp.model.breakdown import return_by_alpha, breakdown_, return_by_episode, episode_by_timestep


def factory(comparison: common.Comparison) -> breakdown_.Breakdown:
    c = common.BreakdownType
    breakdown_type = comparison.breakdown_parameters.breakdown_type
    if breakdown_type == c.EPISODE_BY_TIMESTEP:
        breakdown = episode_by_timestep.EpisodeByTimestep(comparison)
    elif breakdown_type == c.RETURN_BY_EPISODE:
        breakdown = return_by_episode.ReturnByEpisode(comparison)
    elif breakdown_type == c.RETURN_BY_ALPHA:
        breakdown = return_by_alpha.ReturnByAlpha(comparison)
    else:
        raise NotImplementedError

    return breakdown
