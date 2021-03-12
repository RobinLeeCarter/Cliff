from __future__ import annotations
from dataclasses import dataclass

from mdp.model.environment.state import State
from mdp.model.environment.action import Action


@dataclass(frozen=True)
class StateAction:
    state: State
    action: Action
