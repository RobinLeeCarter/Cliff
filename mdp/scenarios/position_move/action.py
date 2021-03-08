from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import common
from mdp.model import environment


@dataclass(frozen=True)
class Action(environment.Action):
    # origin at bottom left
    move: common.XY
