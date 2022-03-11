from __future__ import annotations
from typing import Optional, Callable

from mdp import common
from mdp.model.breakdown.breakdown import Breakdown
from mdp.model.breakdown.return_by_alpha import ReturnByAlpha
from mdp.model.breakdown.rms_by_episode import RmsByEpisode
from mdp.model.breakdown.return_by_episode import ReturnByEpisode
from mdp.model.breakdown.episode_by_timestep import EpisodeByTimestep


def breakdown_factory(comparison: common.Comparison,
                      algorithm_name_fn: Optional[Callable[[common.AlgorithmType], str]] = None) \
        -> Optional[Breakdown]:
    c = common.BreakdownType
    breakdown_type = comparison.breakdown_parameters.breakdown_type
    breakdown: Optional[Breakdown]
    if breakdown_type == c.EPISODE_BY_TIMESTEP:
        breakdown = EpisodeByTimestep(comparison)
    elif breakdown_type == c.RETURN_BY_EPISODE:
        breakdown = ReturnByEpisode(comparison)
    elif breakdown_type == c.RMS_BY_EPISODE:
        breakdown = RmsByEpisode(comparison)
    elif breakdown_type == c.RETURN_BY_ALPHA:
        breakdown = ReturnByAlpha(comparison, algorithm_name_fn)
    else:
        breakdown = None

    return breakdown
