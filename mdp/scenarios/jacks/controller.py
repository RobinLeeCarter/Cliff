from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.scenarios.jacks.model.model import Model

from mdp import controller


class Controller(controller.Controller):
    def __init__(self):
        super().__init__()
        self._model: Optional[Model] = self._model

    def output(self):
        if self._comparison.graph3d_values.show_graph:
            self._model.environment.insert_state_function_into_graph3d(
                comparison=self._comparison,
                v=self._model.agent.algorithm.V
            )
            self._view.graph3d.make_plot(self._comparison.graph3d_values)

        if self._comparison.grid_view_parameters.show_result:
            self._model.environment.update_grid_policy(policy=self._model.agent.policy)
            self._view.grid_view.display_latest_step()
