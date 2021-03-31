from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.scenarios.jacks.model.model import Model
    from mdp.model.agent.agent import Agent
    from mdp.scenarios.jacks.model.environment import Environment

from mdp import controller


class Controller(controller.Controller):
    def __init__(self):
        super().__init__()
        self._model: Optional[Model] = self._model

    def output(self):
        environment: Environment = self._model.environment
        agent: Agent = self._model.agent

        environment.insert_state_function_into_graph3d(
            comparison=self._comparison,
            v=agent.algorithm.V
        )
        self._view.graph3d.make_plot(self._comparison.graph3d_values)

        environment.update_grid_policy(policy=agent.policy)
        self._view.grid_view.display_latest_step()
