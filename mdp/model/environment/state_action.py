from __future__ import annotations

from mdp.model.environment import State, Action

from dataclasses import dataclass


@dataclass(frozen=True)
class StateAction:
    state: State
    action: Action
