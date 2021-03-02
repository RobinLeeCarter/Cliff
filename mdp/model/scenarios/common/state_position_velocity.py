from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import common
from mdp.model.scenarios.common import state_position


@dataclass(frozen=True)
class State(state_position.State):
    # origin at bottom left
    velocity: common.XY
