from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import copy

import numpy as np

from mdp import common

if TYPE_CHECKING:
    from mdp.scenario.jacks.controller import Controller
    from mdp.scenario.blackjack.model.environment_parameters import EnvironmentParameters
from mdp.model.tabular.tabular_model import TabularModel
from mdp.scenario.blackjack.model.environment import Environment
from mdp.scenario.blackjack.model.state import State
from mdp.scenario.blackjack.model.action import Action


class Model(TabularModel[State, Action, Environment]):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self._controller: Optional[Controller] = None

    def _create_environment(self, environment_parameters: EnvironmentParameters) -> Environment:
        return Environment(environment_parameters)

    def get_state_function_graph(self, usable_ace: bool) -> common.Graph3DValues:
        env: Environment = self.environment

        x_values = np.array(env.player_sums, dtype=int)
        y_values = np.array(env.dealers_cards, dtype=int)
        z_values = np.empty(shape=y_values.shape + x_values.shape, dtype=float)

        for player_sum in env.player_sums:
            for dealers_card in env.dealers_cards:
                state: State = State(
                    is_terminal=False,
                    player_sum=player_sum,
                    usable_ace=usable_ace,
                    dealers_card=dealers_card,
                )
                x = player_sum - env.player_sum_min
                y = dealers_card - env.dealers_card_min
                s = env.state_index[state]
                z_values[y, x] = self.algorithm.V[s]
                # print(player_sum, dealer_card, v[state])

        g: common.Graph3DValues = copy.copy(self._comparison.graph3d_values)
        if usable_ace:
            g.title = "Usable Ace"
        else:
            g.title = "No usable Ace"
        g.x_series = common.Series(title=g.x_label, values=x_values)
        g.y_series = common.Series(title=g.y_label, values=y_values)
        g.z_series = common.Series(title=g.z_label, values=z_values)
        return g
