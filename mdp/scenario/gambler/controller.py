from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.scenario.gambler.model.model import Model

from mdp import controller


class Controller(controller.Controller):
    def __init__(self):
        super().__init__()
        self._model: Optional[Model] = self._model

    def output(self):
        self._model.environment.insert_state_function_into_graph2d(
            self._comparison,
            self._model.agent.algorithm.V
        )
        self._view.graph.make_plot(self._comparison.graph_values)

        self._model.environment.insert_policy_into_graph2d(
            self._comparison,
            self._model.agent.policy
        )
        self._view.graph.make_plot(self._comparison.graph_values)
