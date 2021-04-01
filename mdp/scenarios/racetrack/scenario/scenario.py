from __future__ import annotations
from abc import ABC

from mdp.scenarios import scenario
# from mdp.scenarios.racetrack.model.environment_parameters import EnvironmentParameters
from mdp.scenarios.racetrack.model.model import Model
from mdp.scenarios.racetrack.controller import Controller
from mdp.scenarios.racetrack.view.view import View


class Scenario(scenario.Scenario, ABC):
    def _create_model(self) -> Model:
        return Model()

    def _create_controller(self) -> Controller:
        return Controller()

    def _create_view(self) -> View:
        return View()
