from __future__ import annotations
from dataclasses import dataclass
from abc import ABC

from mdp.model.base.environment.base_action import BaseAction


@dataclass(frozen=True)
class TabularAction(BaseAction, ABC):
    pass
