from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from mdp.model.tabular.policy.tabular_policy import TabularPolicy
    from mdp.scenario.gambler.controller import Controller
    from mdp.scenario.gambler.comparison.environment_parameters import EnvironmentParameters
from mdp import common
from mdp.model.tabular.tabular_model import TabularModel
from mdp.scenario.gambler.model.environment import Environment
from mdp.scenario.gambler.model.state import State
from mdp.scenario.gambler.model.action import Action


class Model(TabularModel[State, Action, Environment]):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self._controller: Optional[Controller] = None
        self.environment: Optional[Environment] = None

    def _create_environment(self, environment_parameters: EnvironmentParameters) -> Environment:
        return Environment(environment_parameters)

    def get_state_graph_values(self) -> common.Graph2DValues:
        x_list: list[int] = []
        y_list: list[float] = []
        for s, state in enumerate(self.environment.states):
            if not state.is_terminal:
                x_list.append(state.capital)
                y_list.append(self.algorithm.V[s])
                # print(state.capital, v[state])
        x_values = np.array(x_list, dtype=int)
        y_values = np.array(y_list, dtype=float)

        g: common.Graph2DValues = common.Graph2DValues()
        g.x_series = common.Series(title=g.x_label, values=x_values)
        g.graph_series = [common.Series(title=g.y_label, values=y_values)]
        g.title = "V(s)"
        g.x_label = "Capital"
        g.y_label = "V(s)"
        g.x_min = 0.0
        g.x_max = 100.0
        g.y_min = 0.0
        g.y_max = 1.0
        g.has_grid = True
        g.has_legend = False
        return g

    def get_policy_graph_values(self) -> common.Graph2DValues:
        policy: TabularPolicy = self.algorithm.target_policy

        x_list: list[int] = []
        y_list: list[float] = []
        for s, state in enumerate(self.environment.states):
            if not state.is_terminal:
                x_list.append(state.capital)
                action: Action = policy.get_action(s)   # type: ignore
                y_list.append(float(action.stake))
                # print(state.capital, v[state])
        x_values = np.array(x_list, dtype=int)
        y_values = np.array(y_list, dtype=float)

        g: common.Graph2DValues = common.Graph2DValues()
        g.x_series = common.Series(title=g.x_label, values=x_values)
        g.graph_series = [common.Series(title=g.y_label, values=y_values)]
        g.title = "Policy"
        g.x_label = "Capital"
        g.y_label = "Stake"
        g.x_min = 0.0
        g.x_max = 100.0
        g.y_min = 0.0
        g.y_max = None
        g.has_grid = True
        g.has_legend = False
        return g
