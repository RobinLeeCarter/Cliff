from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from breakdown import breakdown_
    from view import graph
import common
from breakdown import episode_by_timestep, return_by_alpha, return_by_episode


def factory(scenario: common.Scenario, graph_: graph.Graph) -> breakdown_.Breakdown:
    c = common.BreakdownType
    breakdown_type = scenario.breakdown_parameters.breakdown_type
    if breakdown_type == c.EPISODE_BY_TIMESTEP:
        breakdown = episode_by_timestep.EpisodeByTimestep(scenario, graph_)
    elif breakdown_type == c.RETURN_BY_EPISODE:
        breakdown = return_by_episode.ReturnByEpisode(scenario, graph_)
    elif breakdown_type == c.RETURN_BY_ALPHA:
        breakdown = return_by_alpha.ReturnByAlpha(scenario, graph_)
    else:
        raise NotImplementedError

    return breakdown
