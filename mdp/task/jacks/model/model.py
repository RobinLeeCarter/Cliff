from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

from mdp import common
from mdp.model.tabular.policy.tabular_policy import TabularPolicy

if TYPE_CHECKING:
    from mdp.task.jacks.controller import Controller
    from mdp.model.tabular.agent.tabular_episode import TabularEpisode
    from mdp.task.jacks.model.environment_parameters import EnvironmentParameters

from mdp.model.tabular.tabular_model import TabularModel
from mdp.task.jacks.model.environment import Environment
from mdp.task.jacks.model.state import State
from mdp.task.jacks.model.action import Action


class Model(TabularModel[State, Action, Environment]):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self._controller: Optional[Controller] = None
        self.environment: Optional[Environment] = None

    def _create_environment(self, environment_parameters: EnvironmentParameters) -> Environment:
        return Environment(environment_parameters)

    def _display_step(self, episode: Optional[TabularEpisode]):
        policy = self.algorithm.target_policy
        assert isinstance(policy, TabularPolicy)
        self.environment.update_grid_policy(policy)
        self._controller.display_step(episode)

    def get_state_graph3d_values(self) -> common.Graph3DValues:
        max_cars: int = self.environment.max_cars
        x_values = np.arange(max_cars + 1, dtype=float)
        y_values = np.arange(max_cars + 1, dtype=float)
        z_values = np.empty(shape=(max_cars + 1, max_cars + 1), dtype=float)

        for cars1 in range(max_cars+1):
            for cars2 in range(max_cars+1):
                state: State = State(
                    ending_cars_1=cars1,
                    ending_cars_2=cars2,
                    is_terminal=False,
                )
                s: int = self.environment.state_index[state]
                z_values[cars2, cars1] = self.algorithm.V[s]
                # print(cars1, cars2, v[state])

        g: common.Graph3DValues = self._comparison.graph3d_values
        g.x_series = common.Series(title=g.x_label, values=x_values)
        g.y_series = common.Series(title=g.y_label, values=y_values)
        g.z_series = common.Series(title=g.z_label, values=z_values)
        return g
