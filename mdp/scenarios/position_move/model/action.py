from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import common
from mdp.model.environment import action


@dataclass(frozen=True)
class Action(action.Action):
    # origin at bottom left
    move: common.XY
