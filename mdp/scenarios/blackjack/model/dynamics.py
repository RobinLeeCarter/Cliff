from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mdp.scenarios.blackjack.model.action import Action
    from mdp.scenarios.blackjack.model.environment import Environment
    from mdp.scenarios.blackjack.model.environment_parameters import EnvironmentParameters

from mdp.common import Distribution
from mdp.model.environment import dynamics

from mdp.scenarios.blackjack.model.state import State
from mdp.scenarios.blackjack.model.response import Response
from mdp.scenarios.blackjack.model.enums import Result


class Dynamics(dynamics.Dynamics):
    def __init__(self, environment_: Environment, environment_parameters: EnvironmentParameters):
        super().__init__(environment_, environment_parameters)

        # downcast
        self._environment: Environment = self._environment
        self._max_card: int = 10    # 10, J, Q or K combined
        self._card_distribution: Distribution[int] = Distribution()
        self._start_distribution: Distribution[State] = Distribution()

    def build(self):
        thirteenth: float = 1.0 / 13.0  # 13 cards in a suit
        self._card_distribution = Distribution[int]({c: thirteenth for c in range(1, self._max_card)})
        self._card_distribution[self._max_card] = 4.0 * thirteenth     # 10, J, Q or K
        self._card_distribution.enable()

        start_states: list[State] = [state for state in self._environment.states if not state.is_terminal]
        start_p: float = 1 / len(start_states)
        self._start_distribution = Distribution[State]({state: start_p for state in start_states})
        self._start_distribution.enable()

        super().build()

    def get_a_start_state(self) -> State:
        return self._start_distribution.draw_one()

    def draw_response(self, state: State, action: Action) -> Response:
        """
        draw a single outcome for a single state and action
        standard call for episodic algorithms
        """
        player_sum: int = state.player_sum
        usable_ace: bool = state.usable_ace
        result: Optional[Result] = None
        dealers_sum: int = state.dealers_card

        if action.hit:
            card = self._card_distribution.draw_one()
            if self._verbose:
                print(f"player card = {card}")
            # once player sum is 12 an extra ace can never be 'usable' so just counts as 1
            player_sum += card

            if player_sum > 21 and usable_ace:
                # use ace
                player_sum -= 10
                usable_ace = False

            if player_sum > 21:
                result = Result.LOSE
        else:
            # dealers_turn
            dealers_sum = self.dealers_turn(state.dealers_card)
            result = self.get_result(player_sum, dealers_sum)

        if self._verbose:
            print(f"player_sum = {player_sum}")
            print(f"dealers_sum = {dealers_sum}")
            print(f"result = {result}")
        if result is None:
            # continue players turn
            return Response(
                reward=0.0,
                state=State(
                    is_terminal=False,
                    player_sum=player_sum,
                    usable_ace=usable_ace,
                    dealers_card=state.dealers_card
                )
            )
        if result == result.LOSE:
            return Response(reward=-1.0, state=State(is_terminal=True, result=-1))
        elif result == result.DRAW:
            return Response(reward=0.0, state=State(is_terminal=True, result=0))
        elif result == result.WIN:
            return Response(reward=1.0, state=State(is_terminal=True, result=1))
        else:
            raise NotImplementedError

    def dealers_turn(self, starting_card: int) -> int:
        dealers_sum: int
        dealers_usable_ace: bool
        if starting_card == 1:
            dealers_sum = 11
            dealers_usable_ace = True
        else:
            dealers_sum = starting_card
            dealers_usable_ace = False

        while dealers_sum < 17:
            card = self._card_distribution.draw_one()
            if self._verbose:
                print(f"dealers card = {card}")
            if card == 1 and not dealers_usable_ace and dealers_sum + 11 <= 21:
                # drawn an ace and is usable
                dealers_usable_ace = True
                dealers_sum += 11
            else:
                dealers_sum += card
                if dealers_sum > 21 and dealers_usable_ace:
                    # use ace - switch from counting as 11 to counting as 1
                    dealers_sum -= 10

        return dealers_sum

    def get_result(self, player_sum, dealers_sum) -> Result:
        if dealers_sum > 21:
            return Result.WIN
        elif dealers_sum > player_sum:
            return Result.LOSE
        elif dealers_sum == player_sum:
            return Result.DRAW
        else:
            return Result.WIN
