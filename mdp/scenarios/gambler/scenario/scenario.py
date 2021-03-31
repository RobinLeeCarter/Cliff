from __future__ import annotations
from abc import ABC

from mdp import common, scenario
from mdp.scenarios.gambler.model.environment_parameters import EnvironmentParameters
from mdp.scenarios.gambler.model.model import Model
from mdp.scenarios.gambler.controller import Controller
# from mdp.scenarios.gambler.view.view import View


class Scenario(scenario.Scenario, ABC):
    def __init__(self):
        super().__init__()
        self._environment_parameters = EnvironmentParameters(
            environment_type=common.ScenarioType.GAMBLER,
            probability_heads=0.4,
        )

    def _create_model(self) -> Model:
        return Model()

    def _create_controller(self) -> Controller:
        return Controller()

    # def _create_view(self) -> View:
    #     return View()
