from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.agent.agent import Agent
    from mdp.scenarios.blackjack.model.environment import Environment

from mdp import controller
from mdp.scenarios.blackjack.model.model import Model
from mdp.scenarios.blackjack.view.view import View


class Controller(controller.Controller):
    def __init__(self):
        super().__init__()
        self._model: Optional[Model] = self._model
        self._view: Optional[View] = self._view

    def output(self):
        environment: Environment = self._model.environment
        agent: Agent = self._model.agent

        for usable_ace in [False, True]:
            environment.update_grid_policy_ace(agent.policy, usable_ace)
            self._view.grid_view.set_title(usable_ace)
            self._view.grid_view.display_latest_step()

            environment.insert_state_function_into_graph3d_ace(
                comparison=self._comparison,
                v=agent.algorithm.V,
                usable_ace=usable_ace
            )
            self._view.graph3d.make_plot(self._comparison.graph3d_values)
