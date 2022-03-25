from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.task.jacks.model.action import Action
from mdp.task.jacks.model.model import Model
from mdp.task.jacks.view.view import View

from mdp import common
from mdp.controller.tabular_controller import TabularController


class Controller(TabularController[Model, View]):
    def __init__(self):
        super().__init__()
        self._model: Optional[Model] = self._model
        self._view: Optional[View] = self._view

    def build(self, comparison: common.Comparison):
        super().build(comparison)
        self._view.grid_view.set_gridworld(self._model.environment.grid_world)

    def output(self):
        if self._comparison.graph3d_values:
            graph3d_values: common.Graph3DValues = self._model.get_state_graph3d_values()
            self._view.graph3d.make_plot(graph3d_values)

        grid_view_parameters = self._comparison.grid_view_parameters
        if grid_view_parameters and grid_view_parameters.show_result:
            self._model.environment.update_grid_policy(policy=self._model.algorithm.target_policy)
            self._view.grid_view.display_latest_step()

        policy = self._model.algorithm.target_policy
        total_transfers: int = 0
        for s, state in enumerate(self._model.environment.states):
            if not state.is_terminal:
                action: Action = policy.get_action(s)
                total_transfers += action.transfer_1_to_2
        v: np.ndarray = self._model.algorithm.V.vector
        total_v: float = v.sum()

        print(f"total_transfers: {total_transfers}")
        print(f"total_v: {total_v:.0f}")
