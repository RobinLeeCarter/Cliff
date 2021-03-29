from __future__ import annotations
from dataclasses import dataclass

from mdp.model.environment import action


@dataclass(frozen=True)
class Action(action.Action):
    transfer_1_to_2: int
