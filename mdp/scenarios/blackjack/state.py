from __future__ import annotations
from dataclasses import dataclass

from mdp.model import environment


@dataclass(frozen=True)
class State(environment.State):
    player_sum: int = 0         # 12-21
    usable_ace: bool = False    # player has usable ace?
    dealers_card: int = 0       # Ace-10, Ace represented as 1 for simplicity and graphs
    result: int = 0             # 1 = win, -1 = lose, 0 = draw or still-playing
