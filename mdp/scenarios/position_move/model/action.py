from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import common
from mdp.model.environment.tabular.tabular_action import TabularAction


@dataclass(frozen=True)
class Action(TabularAction):
    # origin at bottom left
    move: common.XY
