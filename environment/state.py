from dataclasses import dataclass

import common


@dataclass(frozen=True)
class State:
    # origin at bottom left (max rows)

    # position
    position: common.XY
    # x: int          # >= 0
    # y: int          # >= 0

    # terminal
    is_terminal: bool = False

    @property
    def index(self) -> tuple[int, int]:
        return self.position.x, self.position.y
