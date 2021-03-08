from __future__ import annotations
from dataclasses import dataclass

from mdp.model import environment


@dataclass(frozen=True)
class Action(environment.Action):
    transfer_1_to_2: int
