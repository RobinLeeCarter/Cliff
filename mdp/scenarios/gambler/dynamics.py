from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mdp.scenarios.gambler.action import Action
    from mdp.scenarios.gambler.environment import Environment
    from mdp.scenarios.gambler.environment_parameters import EnvironmentParameters

import random

from mdp.common import Distribution
from mdp.model import environment

from mdp.scenarios.gambler.state import State
from mdp.scenarios.gambler.response import Response
from mdp.scenarios.gambler.enums import Toss, Result


class Dynamics(environment.Dynamics):
    def __init__(self, environment_: Environment, environment_parameters: EnvironmentParameters):
        super().__init__(environment_, environment_parameters)

        # downcast
        self._environment: Environment = self._environment
        self._environment_parameters: EnvironmentParameters = self._environment_parameters
        # self._probability_heads: float = self._environment_parameters.probability_heads
        self._toss_distribution: Distribution[Toss] = Distribution()

    def build(self):
        ph = self._environment_parameters.probability_heads
        self._toss_distribution[Toss.HEADS] = ph
        self._toss_distribution[Toss.TAILS] = 1.0 - ph
        self._toss_distribution.self_check()
        super().build()

    def get_a_start_state(self) -> State:
        return random.choice([state for state in self._environment.states if not state.is_terminal])

    def draw_response(self, state: State, action: Action) -> Response:
        """
        draw a single outcome for a single state and action
        standard call for episodic algorithms
        """
        toss: Toss = self._toss_distribution.draw_one()
        if toss == Toss.HEADS:
            new_capital = state.capital + action.stake * 2.0
        else:
            new_capital = state.capital - action.stake

        result: Optional[Result] = None
        reward: float = 0.0

        if new_capital == self._environment_parameters.max_capital:
            result = Result.WIN
            reward = 1.0
        elif new_capital == 0:
            result = Result.LOSE

        if self._verbose:
            print(f"starting_capital = {state.capital}")
            print(f"stake = {action.stake}")
            print(f"toss = {toss}")
            print(f"new_capital = {new_capital}")
            print(f"result = {result}")

        is_terminal: bool = (result is not None)
        return Response(reward=reward, state=State(is_terminal=is_terminal, capital=new_capital))
