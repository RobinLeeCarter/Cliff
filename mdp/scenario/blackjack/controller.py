from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.scenario.blackjack.model.model import Model
    from mdp.scenario.blackjack.view.view import View

from mdp import common
from mdp import controller


class Controller(controller.Controller):
    def __init__(self):
        super().__init__()
        self._model: Optional[Model] = self._model
        self._view: Optional[View] = self._view

    def build(self, comparison: common.Comparison):
        super().build(comparison)
        self._view.grid_view.set_gridworld(self._model.environment.grid_world)

    def output(self):
        for usable_ace in [False, True]:
            self._model.environment.update_grid_policy_ace(self._model.agent.policy,
                                                           self._model.agent.algorithm,
                                                           usable_ace)
            self._view.grid_view.set_title(usable_ace)
            self._view.grid_view.display_latest_step()

            self._model.environment.insert_state_function_into_graph3d_ace(
                comparison=self._comparison,
                v=self._model.agent.algorithm.V,
                usable_ace=usable_ace
            )
            self._view.graph3d.make_plot(self._comparison.graph3d_values)
