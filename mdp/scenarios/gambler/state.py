from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from mdp.model import environment


@dataclass(frozen=True)
class State(environment.State):
    capital: int             # 0-100
