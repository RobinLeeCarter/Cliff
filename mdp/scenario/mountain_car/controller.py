from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.scenario.mountain_car.model.model import Model
    from mdp.scenario.mountain_car.view.view import View

# from mdp import common
from mdp.general_controller import GeneralController


class Controller(GeneralController):
    def __init__(self):
        super().__init__()
        self._model: Optional[Model] = self._model
        self._view: Optional[View] = self._view

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
