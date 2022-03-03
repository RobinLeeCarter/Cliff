from __future__ import annotations
from dataclasses import dataclass
from abc import ABC

from mdp.model.environment.general.general_state import GeneralState


@dataclass(frozen=True)
class TabularState(GeneralState, ABC):
    pass
