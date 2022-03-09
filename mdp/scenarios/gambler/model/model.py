from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.scenarios.gambler.controller import Controller
    from mdp.scenarios.gambler.model.environment_parameters import EnvironmentParameters

from mdp.model.tabular.tabular_model import TabularModel
from mdp.scenarios.gambler.model.environment import Environment


class Model(TabularModel):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self._controller: Optional[Controller] = self._controller
        self.environment: Optional[Environment] = self.environment

    def _create_environment(self, environment_parameters: EnvironmentParameters):
        self.environment: Environment = Environment(environment_parameters)
