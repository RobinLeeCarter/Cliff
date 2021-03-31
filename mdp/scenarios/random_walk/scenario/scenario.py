from __future__ import annotations
from abc import ABC

from mdp import scenario
# from mdp.scenarios.random_walk.model.environment_parameters import EnvironmentParameters
from mdp.scenarios.random_walk.model.model import Model
from mdp.scenarios.random_walk.controller import Controller
from mdp.scenarios.random_walk.view.view import View


class Scenario(scenario.Scenario, ABC):
    # def __init__(self):
    #     super().__init__()
    #     self._environment_parameters = EnvironmentParameters(
    #         environment_type=common.ScenarioType.RANDOM_WALK,
    #     )

    def _create_model(self) -> Model:
        return Model()

    def _create_controller(self) -> Controller:
        return Controller()

    def _create_view(self) -> View:
        return View()
