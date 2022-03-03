from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import common
from mdp.model.environment.general import state


@dataclass(frozen=True)
class State(state.State):
    # origin at bottom left
    position: common.XY
