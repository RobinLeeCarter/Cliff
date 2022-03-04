from __future__ import annotations
from dataclasses import dataclass
from abc import ABC


@dataclass(frozen=True)
class GeneralState(ABC):
    is_terminal: bool
