from __future__ import annotations
from dataclasses import dataclass
from abc import ABC

from mdp.model.general.environment.general_state import GeneralState


@dataclass(frozen=True)
class TabularState(GeneralState, ABC):
    pass
