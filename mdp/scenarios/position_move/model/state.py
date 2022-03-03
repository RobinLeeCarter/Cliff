from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import common
from mdp.model.environment.tabular.tabular_state import TabularState


@dataclass(frozen=True)
class State(TabularState):
    # origin at bottom left
    position: common.XY
