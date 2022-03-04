from __future__ import annotations
from dataclasses import dataclass
from abc import ABC

from mdp.model.general.environment.general_action import GeneralAction


@dataclass(frozen=True)
class TabularAction(GeneralAction, ABC):
    pass
