from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.scenario.racetrack.controller import Controller
    from mdp.scenario.racetrack.model.environment_parameters import EnvironmentParameters
from mdp.model.tabular.tabular_model import TabularModel
from mdp.scenario.racetrack.model.environment import Environment
from mdp.scenario.racetrack.model.state import State
from mdp.scenario.racetrack.model.action import Action


class Model(TabularModel[State, Action, Environment]):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self._controller: Optional[Controller] = None

    def _create_environment(self, environment_parameters: EnvironmentParameters) -> Environment:
        return Environment(environment_parameters)
