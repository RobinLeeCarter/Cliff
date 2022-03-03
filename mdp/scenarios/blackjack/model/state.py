from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from mdp.model.environment.tabular.tabular_state import TabularState


@dataclass(frozen=True)
class State(TabularState):
    player_sum: int = 0             # 12-21
    usable_ace: bool = False        # player has usable ace?
    dealers_card: int = 0           # Ace-10, Ace represented as 1 for simplicity and graphs
    result: Optional[int] = None    # 1 = win, -1 = lose, 0 = draw, None = still-playing
