from __future__ import annotations
from dataclasses import dataclass
from abc import ABC

from mdp.model.base.environment.base_state import BaseState


@dataclass(frozen=True)
class TabularState(BaseState, ABC):
    pass
