from __future__ import annotations
from abc import ABC

from mdp import common
from mdp.scenario.general_scenario import GeneralScenario
from mdp.scenario.blackjack.model.environment_parameters import EnvironmentParameters
from mdp.scenario.blackjack.controller import Controller
from mdp.scenario.blackjack.model.model import Model
from mdp.scenario.blackjack.view.view import View


class Scenario(GeneralScenario[Model, View, Controller], ABC):
    def __init__(self):
        super().__init__()
        self._environment_parameters = EnvironmentParameters(
            environment_type=common.ScenarioType.BLACKJACK,
        )
        self._comparison_settings = common.Settings(
            gamma=1.0,
            runs=1,
            training_episodes=500_000,
            episode_print_frequency=10_000,
            policy_parameters=common.PolicyParameters(
                policy_type=common.PolicyType.DETERMINISTIC,
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

    def _create_model(self) -> Model:
        return Model()

    def _create_view(self) -> View:
        return View()

    def _create_controller(self) -> Controller:
        return Controller()
