from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.scenarios.racetrack.controller import Controller
    from mdp.scenarios.racetrack.model.environment_parameters import EnvironmentParameters

from mdp.model import model

from mdp.scenarios.racetrack.model.environment import Environment


class Model(model.Model):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self._controller: Optional[Controller] = self._controller
        self.environment: Optional[Environment] = self.environment

    def _create_environment(self, environment_parameters: EnvironmentParameters) -> Environment:
        return Environment(environment_parameters)
