from __future__ import annotations
from abc import ABC

from mdp import common
from mdp.scenario.general_comparison_builder import GeneralComparisonBuilder
from mdp.scenario.jacks.comparison.environment_parameters import EnvironmentParameters


class ComparisonBuilder(GeneralComparisonBuilder, ABC):
    def __init__(self):
        super().__init__()
        self._max_cars: int = 20      # problem statement = 20
        self._environment_parameters = EnvironmentParameters(
            max_cars=self._max_cars,
            extra_rules=True,  # change this for extra rules in book as per challenge
        )
        self._comparison_settings = common.Settings(
            gamma=0.9,
            policy_parameters=common.PolicyParameters(
                policy_type=common.PolicyType.TABULAR_DETERMINISTIC,
            ),
            algorithm_parameters=common.AlgorithmParameters(
                theta=0.1  # accuracy of policy_evaluation
            ),
            display_every_step=True,
        )
        self._graph3d_values = common.Graph3DValues(
            show_graph=True,
            x_label="Cars at 1st location",
            y_label="Cars at 2nd location",
            z_label="V(s)",
            x_min=0,
            x_max=self._max_cars,
            y_min=0,
            y_max=self._max_cars,
        )
        self._grid_view_parameters = common.GridViewParameters(
            grid_view_type=common.GridViewType.JACKS,
            show_result=True,
            show_policy=True,
        )
