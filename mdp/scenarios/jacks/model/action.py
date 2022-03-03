from __future__ import annotations
from dataclasses import dataclass

from mdp.model.environment.tabular.tabular_action import TabularAction


@dataclass(frozen=True)
class Action(TabularAction):
    transfer_1_to_2: int
