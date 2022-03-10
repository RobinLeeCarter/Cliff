from __future__ import annotations
from abc import ABC

from mdp.scenario.general_scenario import GeneralScenario
# from mdp.scenarios.racetrack.model.environment_parameters import EnvironmentParameters
from mdp.scenario.racetrack.model.model import Model
from mdp.scenario.racetrack.controller import Controller
from mdp.scenario.racetrack.view.view import View


class Scenario(GeneralScenario, ABC):
    def _create_model(self) -> Model:
        return Model()

    def _create_controller(self) -> Controller:
        return Controller()

    def _create_view(self) -> View:
        return View()
