from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mdp.scenario.blackjack.model.environment import Environment
    from mdp.scenario.blackjack.model.environment_parameters import EnvironmentParameters

from mdp.common import Multinoulli
from mdp.scenario.blackjack.model.state import State
from mdp.scenario.blackjack.model.action import Action
from mdp.scenario.blackjack.model.enums import Result

from mdp.model.tabular.environment.tabular_dynamics import TabularDynamics


class Dynamics(TabularDynamics[State, Action]):
    def __init__(self, environment: Environment, environment_parameters: EnvironmentParameters):
        super().__init__(environment, environment_parameters)
        self._environment: Environment = environment
        self._environment_parameters: EnvironmentParameters = environment_parameters

        self._max_card: int = 10    # 10, J, Q or K combined
        self._card_distribution: Multinoulli[int] = Multinoulli()

    def build(self):
        p: float = 1.0 / 13.0  # 13 cards in a suit
        self._card_distribution = Multinoulli[int]({c: p for c in range(1, self._max_card)})
        self._card_distribution[self._max_card] = 4.0 * p     # 10, J, Q or K
        self._card_distribution.enable()

        super().build()

    def get_start_states(self) -> list[State]:
        return [state for state in self._environment.states if not state.is_terminal]

    def draw_response(self, state: State, action: Action) -> tuple[float, State]:
        """
        draw a single outcome for a single state and action
        standard call for episodic algorithms
        """
        player_sum: int = state.player_sum
        usable_ace: bool = state.usable_ace
        result: Optional[Result] = None
        dealers_sum: int = state.dealers_card
        reward: float
        new_state: State

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
            reward = 0.0
            new_state = State(
                is_terminal=False,
                player_sum=player_sum,
                usable_ace=usable_ace,
                dealers_card=state.dealers_card
            )
        elif result == Result.LOSE:
            reward = -1.0
            new_state = State(is_terminal=True, result=-1)
        elif result == Result.DRAW:
            reward = 0.0
            new_state = State(is_terminal=True, result=0)
        elif result == Result.WIN:
            reward = 1.0
            new_state = State(is_terminal=True, result=1)
        else:
            raise NotImplementedError

        return reward, new_state

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
