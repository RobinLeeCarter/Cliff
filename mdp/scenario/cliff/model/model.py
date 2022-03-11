from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.scenario.cliff.controller import Controller
    from mdp.scenario.cliff.model.environment_parameters import EnvironmentParameters

from mdp.scenario.position_move.model import model
from mdp.scenario.cliff.model.environment import Environment


class Model(model.Model[Environment]):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self._controller: Optional[Controller] = None

    def _create_environment(self, environment_parameters: EnvironmentParameters) -> Environment:
        return Environment(environment_parameters)
