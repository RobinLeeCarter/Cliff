from __future__ import annotations
from abc import ABC

from mdp.scenario.general_scenario import GeneralScenario
from mdp.scenario.gambler.model.model import Model
from mdp.scenario.gambler.controller import Controller
from mdp.scenario.gambler.view.view import View


class Scenario(GeneralScenario[Model, View, Controller], ABC):
    def _create_model(self) -> Model:
        return Model()

    def _create_view(self) -> View:
        return View()

    def _create_controller(self) -> Controller:
        return Controller()
