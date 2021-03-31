from __future__ import annotations
from dataclasses import dataclass

from mdp.model.environment import action


@dataclass(frozen=True)
class Action(action.Action):
    hit: bool   # else 'stick'
