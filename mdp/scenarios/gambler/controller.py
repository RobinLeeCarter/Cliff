from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.agent.agent import Agent
    from mdp.scenarios.gambler.model.environment import Environment
    from mdp.view.graph import Graph

# from mdp import common
from mdp import controller
from mdp.scenarios.gambler.model.model import Model
# from mdp.scenarios.gambler.view.view import View


class Controller(controller.Controller):
    def __init__(self):
        super().__init__()
        self._model: Optional[Model] = self._model
        # self._view: Optional[View] = self._view

    def output(self):
        environment: Environment = self._model.environment
        agent: Agent = self._model.agent
        graph: Graph = self._view.graph

        environment.insert_state_function_into_graph2d(
            self._comparison,
            agent.algorithm.V
        )
        graph.make_plot(self._comparison.graph_values)

        environment.insert_policy_into_graph2d(
            self._comparison,
            agent.policy
        )
        graph.make_plot(self._comparison.graph_values)
