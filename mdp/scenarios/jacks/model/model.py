from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.agent.episode import Episode
    from mdp.scenarios.jacks.model.environment_parameters import EnvironmentParameters

from mdp.model import model
# from mdp.scenarios.jacks.controller import Controller
from mdp.scenarios.jacks.model.environment import Environment


class Model(model.Model):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        # self._controller: Optional[Controller] = self._controller
        self.environment: Optional[Environment] = self.environment

    def _create_environment(self, environment_parameters: EnvironmentParameters) -> Environment:
        return Environment(environment_parameters)

    def _display_step(self, episode_: Optional[Episode]):
        self.environment.update_grid_policy(policy=self.agent.policy)
        self._controller.display_step(episode_)
