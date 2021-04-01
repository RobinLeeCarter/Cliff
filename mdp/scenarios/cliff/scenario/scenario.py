from __future__ import annotations
from abc import ABC

from mdp import scenario
from mdp.scenarios.cliff.controller import Controller
from mdp.scenarios.cliff.model.model import Model
from mdp.scenarios.cliff.view.view import View


class Scenario(scenario.Scenario, ABC):
    def _create_model(self) -> Model:
        return Model()

    def _create_view(self) -> View:
        return View()

    def _create_controller(self) -> Controller:
        return Controller()
