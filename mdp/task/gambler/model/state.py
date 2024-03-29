from __future__ import annotations
from dataclasses import dataclass

from mdp.model.tabular.environment.tabular_state import TabularState


@dataclass(frozen=True)
class State(TabularState):
    capital: int             # 0-100
