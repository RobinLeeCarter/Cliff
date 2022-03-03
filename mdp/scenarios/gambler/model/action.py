from __future__ import annotations
from dataclasses import dataclass

from mdp.model.environment.general import action


@dataclass(frozen=True)
class Action(action.Action):
    stake: int
