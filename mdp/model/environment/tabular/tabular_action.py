from __future__ import annotations
from dataclasses import dataclass
from abc import ABC

from mdp.model.environment.general.general_action import GeneralAction


@dataclass(frozen=True)
class TabularAction(GeneralAction, ABC):
    pass
