from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.scenario.jacks.controller import Controller
    from mdp.model.tabular.agent.episode import Episode
    from mdp.scenario.jacks.model.environment_parameters import EnvironmentParameters

from mdp.model.tabular.tabular_model import TabularModel
from mdp.scenario.jacks.model.environment import Environment


class Model(TabularModel[Environment]):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self._controller: Optional[Controller] = self._controller

    def _create_environment(self, environment_parameters: EnvironmentParameters) -> Environment:
        return Environment(environment_parameters)

    def _display_step(self, episode_: Optional[Episode]):
        self.environment.update_grid_policy(policy=self.agent.policy)
        self._controller.display_step(episode_)
