from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.scenario.gambler.controller import Controller
    from mdp.scenario.gambler.comparison.environment_parameters import EnvironmentParameters

from mdp.model.tabular.tabular_model import TabularModel
from mdp.scenario.gambler.model.environment import Environment
from mdp.scenario.gambler.model.state import State
from mdp.scenario.gambler.model.action import Action


class Model(TabularModel[State, Action, Environment]):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self._controller: Optional[Controller] = None
        self.environment: Optional[Environment] = None

    def _create_environment(self, environment_parameters: EnvironmentParameters) -> Environment:
        return Environment(environment_parameters)
