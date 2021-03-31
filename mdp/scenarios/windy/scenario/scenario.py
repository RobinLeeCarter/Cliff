from __future__ import annotations
from abc import ABC

from mdp import common, scenario
from mdp.scenarios.windy.model.environment_parameters import EnvironmentParameters
from mdp.scenarios.windy.controller import Controller
from mdp.scenarios.windy.model.model import Model
from mdp.scenarios.windy.view.view import View


class Scenario(scenario.Scenario, ABC):
    def __init__(self, random_wind: bool):
        super().__init__()
        self._random_wind: bool = random_wind

    def _create_model(self) -> Model:
        return Model()

    def _create_view(self) -> View:
        return View()

    def _create_controller(self) -> Controller:
        return Controller()
