from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from environment import state


@dataclass(frozen=True)
class Observation:
    reward: Optional[float]
    state: state.State
