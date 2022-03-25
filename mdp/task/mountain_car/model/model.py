from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.task.mountain_car.controller import Controller
    from mdp.task.mountain_car.model.environment_parameters import EnvironmentParameters
from mdp.model.non_tabular.non_tabular_model import NonTabularModel
from mdp.task.mountain_car.model.state import State
from mdp.task.mountain_car.model.action import Action
from mdp.task.mountain_car.model.environment import Environment


class Model(NonTabularModel[State, Action, Environment]):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self._controller: Optional[Controller] = None

    def _create_environment(self, environment_parameters: EnvironmentParameters) -> Environment:
        return Environment(environment_parameters)
