from __future__ import annotations
from abc import ABC

from mdp.scenario.general_scenario import GeneralScenario
# from mdp.scenarios.gambler.model.environment_parameters import EnvironmentParameters
from mdp.scenario.gambler.model.model import Model
from mdp.scenario.gambler.controller import Controller
# from mdp.scenarios.gambler.view.view import View


class Scenario(GeneralScenario, ABC):
    # def __init__(self):
    #     super().__init__()
    #     self._environment_parameters = EnvironmentParameters(
    #         environment_type=common.ScenarioType.GAMBLER,
    #         probability_heads=0.4,
    #     )

    def _create_model(self) -> Model:
        return Model()

    def _create_controller(self) -> Controller:
        return Controller()

    # def _create_view(self) -> View:
    #     return View()
