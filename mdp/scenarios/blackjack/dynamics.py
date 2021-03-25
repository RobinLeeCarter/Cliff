from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import random
from scipy import stats

if TYPE_CHECKING:
    from mdp.scenarios.blackjack.action import Action
    from mdp.scenarios.blackjack.environment import Environment
    from mdp.scenarios.blackjack.environment_parameters import EnvironmentParameters

import random

from mdp.common import Distribution
from mdp.model import environment

from mdp.scenarios.blackjack.state import State
from mdp.scenarios.blackjack.response import Response


class Dynamics(environment.Dynamics):
    def __init__(self, environment_: Environment, environment_parameters: EnvironmentParameters):
        super().__init__(environment_, environment_parameters)

        # downcast
        self._environment: Environment = self._environment
        self._max_card: int = 10    # 10, J, Q or K combined
        self._card_distribution: Distribution[int] = Distribution()

    def build(self):
        thirteenth: float = 1.0 / 13.0  # 13 cards in a suit
        self._card_distribution = Distribution({c: thirteenth for c in range(self._max_card)})
        self._card_distribution[self._max_card] += 4.0 * thirteenth     # 10, J, Q or K
        self._card_distribution.self_check()
        super().build()

    def get_a_start_state(self) -> State:
        return random.choice([state for state in self._environment.states if not state.is_terminal])

    def draw_response(self, state: State, action: Action) -> Response:
        """
        draw a single outcome for a single state and action
        standard call for episodic algorithms
        """
        player_sum: int = state.player_sum
        usable_ace: bool = state.usable_ace
        bust: bool = False

        if action.hit:
            card = self._card_distribution.draw_one()
            # once player sum is 12 an extra ace can never be 'usable' so just count as 1
            player_sum = state.player_sum + card
            if player_sum > 21:
                if usable_ace:
                    # use ace
                    player_sum -= 10
                    usable_ace = False
                else:
                    bust = True
            if bust:
                new_state = State(is_terminal=True, result=-1)
            else:
                new_state = State(is_terminal=False,
                                  player_sum=player_sum,
                                  usable_ace=usable_ace,
                                  dealers_card=state.dealers_card)
        else:
            # TODO: dealers turn
            # dealers turn
            card = self._card_distribution.draw_one()
            # etc...

        new_state = State(is_terminal=False, )
        reward: float
        return Response(reward, new_state)




