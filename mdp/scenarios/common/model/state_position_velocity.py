from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import common
from mdp.scenarios.common.model import state_position


@dataclass(frozen=True)
class StatePositionVelocity(state_position.StatePosition):
    # origin at bottom left
    velocity: common.XY
