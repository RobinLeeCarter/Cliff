from __future__ import annotations
from abc import ABC

from mdp import common
from mdp.scenario.general_comparison_builder import GeneralComparisonBuilder


class ComparisonBuilder(GeneralComparisonBuilder, ABC):
    def __init__(self):
        super().__init__()
        self._comparison_settings = common.Settings(
            gamma=1.0,
            runs=1,
            training_episodes=500_000,
            episode_print_frequency=10_000,
            policy_parameters=common.PolicyParameters(
                policy_type=common.PolicyType.TABULAR_DETERMINISTIC,
            ),
        )
        self._graph3d_values = common.Graph3DValues(
            show_graph=True,
            x_label="Player sum",
            y_label="Dealer showing",
            z_label="V(s)",
            x_min=12,
            x_max=21,
            y_min=1,
            y_max=10,
            z_min=-1.0,
            z_max=1.0,
            multi_parameter=[False, True]
        )
        self._grid_view_parameters = common.GridViewParameters(
            grid_view_type=common.GridViewType.BLACKJACK,
            show_result=True,
            show_policy=True,
            show_q=True,
        )
