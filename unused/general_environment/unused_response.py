from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from mdp.model.base.environment.base_state import BaseState


@dataclass(frozen=True)
class Response:
    reward: Optional[float]
    state: BaseState
