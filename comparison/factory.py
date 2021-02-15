from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comparison import comparison_
    from view import graph
import common
from comparison import episode_by_timestep, return_by_alpha, return_by_episode


def factory(scenario: common.Scenario, graph_: graph.Graph) -> comparison_.Comparison:
    c = common.ComparisonType
    comparison_type = scenario.comparison_parameters.comparison_type
    if comparison_type == c.EPISODE_BY_TIMESTEP:
        comparison = episode_by_timestep.EpisodeByTimestep(scenario, graph_)
    elif comparison_type == c.RETURN_BY_EPISODE:
        comparison = return_by_episode.ReturnByEpisode(scenario, graph_)
    elif comparison_type == c.RETURN_BY_ALPHA:
        comparison = return_by_alpha.ReturnByAlpha(scenario, graph_)
    else:
        raise NotImplementedError

    return comparison
