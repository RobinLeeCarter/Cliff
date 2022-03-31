from __future__ import annotations
from abc import ABC

from mdp import common
from mdp.task.base_comparison_builder import BaseComparisonBuilder


class ComparisonBuilder(BaseComparisonBuilder, ABC):
    def __init__(self):
        super().__init__()
        self._graph2d_values = common.Graph2DValues(
                has_grid=True,
                has_legend=True,
            )
        self._grid_view_parameters = common.GridViewParameters(
                show_demo=True,
                show_q=True,
            )
        self._breakdown_parameters = common.BreakdownParameters(
                breakdown_type=common.BreakdownType.EPISODE_BY_TIMESTEP,
            )
        self._settings = common.Settings(
            runs=1,
            training_episodes=170,
            review_every_step=True,
            algorithm_parameters=common.AlgorithmParameters(
                algorithm_type=common.AlgorithmType.TABULAR_SARSA,
                alpha=0.5,
            )
        )
