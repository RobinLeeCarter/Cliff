from __future__ import annotations

from dataclasses import dataclass

import common
from environment import environment_m


@dataclass(frozen=True)
class Action:
    move: common.XY

    @property
    def index(self) -> tuple[int]:
        return environment_m.Environment.actions_singleton.get_index_from_action(self)
