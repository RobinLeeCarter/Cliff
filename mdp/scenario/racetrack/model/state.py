from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import common
from mdp.scenario.position.model import state


@dataclass(frozen=True)
class State(state.State):
    # origin at bottom left
    velocity: common.XY
