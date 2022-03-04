from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from mdp.model.general.environment.general_state import GeneralState


@dataclass(frozen=True)
class Response:
    reward: Optional[float]
    state: GeneralState
