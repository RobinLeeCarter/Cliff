from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from model.environment import state


@dataclass(frozen=True)
class Response:
    reward: Optional[float]
    state: state.State
