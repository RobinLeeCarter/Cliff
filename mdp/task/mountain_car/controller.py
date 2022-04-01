from __future__ import annotations
from typing import Optional

from mdp.task.mountain_car.model.model import Model
from mdp.task.mountain_car.view.view import View

from mdp import common
from mdp.controller.non_tabular_controller import NonTabularController


class Controller(NonTabularController[Model, View],
                 environment_type=common.EnvironmentType.MOUNTAIN_CAR):
    def __init__(self):
        super().__init__()
        self._model: Optional[Model] = self._model
        self._view: Optional[View] = self._view

    def output(self):
        if self._comparison.graph3d_values:
            graph3d_values: common.Graph3DValues = self._model.get_state_action_graph()
            self._view.graph3d.make_plot(graph3d_values)

    # def build(self, comparison: common.Comparison):
    #     super().build(comparison)
        # self._view.grid_view.set_gridworld(self._model.environment.grid_world)

    # def output(self):
    #     self._breakdown_graph()
    #
    #     # prep for output
    #     self._model.environment.grid_world.skid_probability = 0.0
    #     self._model.switch_to_target_policy()
    #     # output demo
    #     self._view.grid_view.demonstrate(self.new_episode_request)
