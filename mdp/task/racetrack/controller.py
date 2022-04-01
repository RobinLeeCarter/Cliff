from __future__ import annotations
from typing import Optional

from mdp.task.racetrack.model.model import Model
from mdp.task.racetrack.view.view import View

from mdp import common
from mdp.controller.tabular_controller import TabularController


class Controller(TabularController[Model, View],
                 environment_type=common.EnvironmentType.RACETRACK):
    def __init__(self):
        super().__init__()
        self._model: Optional[Model] = self._model
        self._view: Optional[View] = self._view

    def build(self, comparison: common.Comparison):
        super().build(comparison)
        self._view.grid_view.set_gridworld(self._model.environment.grid_world)

    def output(self):
        self._breakdown_2dgraph()

        # prep for output
        self._model.environment.grid_world.skid_probability = 0.0
        self._model.switch_agent_to_target_policy()
        # output demo
        self._view.grid_view.demonstrate(self.new_episode_request)
