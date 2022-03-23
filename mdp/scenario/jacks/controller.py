from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.scenario.jacks.model.action import Action
from mdp.scenario.jacks.model.model import Model
from mdp.scenario.jacks.view.view import View

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
        if self._comparison.graph3d_values.show_graph:
            self._model.environment.insert_state_function_into_graph3d(
                comparison=self._comparison,
                v=self._model.algorithm.V
            )
            self._view.graph3d.make_plot(self._comparison.graph3d_values)

        if self._comparison.grid_view_parameters.show_result:
            self._model.environment.update_grid_policy(policy=self._model.target_policy)
            self._view.grid_view.display_latest_step()

        policy = self._model.target_policy
        total_transfers: int = 0
        for s, state in enumerate(self._model.environment.states):
            if not state.is_terminal:
                action: Action = policy.get_action(s)
                total_transfers += action.transfer_1_to_2
        v: np.ndarray = self._model.algorithm.V.vector
        total_v: float = v.sum()

        print(f"total_transfers: {total_transfers}")
        print(f"total_v: {total_v:.0f}")
