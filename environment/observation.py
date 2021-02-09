from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from environment import state


@dataclass(frozen=True)
class Observation:
    reward: Optional[float]
    state: state.State
