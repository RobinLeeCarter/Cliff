from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from mdp.model.environment.general.state import State


@dataclass(frozen=True)
class Response:
    reward: Optional[float]
    state: State
