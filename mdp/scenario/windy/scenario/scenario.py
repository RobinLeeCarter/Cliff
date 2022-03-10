from __future__ import annotations
from abc import ABC

from mdp.scenario.general_scenario import GeneralScenario
from mdp.scenario.windy.controller import Controller
from mdp.scenario.windy.model.model import Model
from mdp.scenario.windy.view.view import View


class Scenario(GeneralScenario[Model, View, Controller], ABC):
    def __init__(self, random_wind: bool = False):
        super().__init__()
        self._random_wind: bool = random_wind

    def _create_model(self) -> Model:
        return Model()

    def _create_view(self) -> View:
        return View()

    def _create_controller(self) -> Controller:
        return Controller()
