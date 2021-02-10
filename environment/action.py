from __future__ import annotations

from dataclasses import dataclass

import common


@dataclass(frozen=True)
class Action:
    move: common.XY

    @property
    def index(self) -> tuple[int]:
        return actions.get_index_from_action(self)


class Actions:
    def __init__(self):
        self.action_list: list[Action] = []
        self._build_action_list()

        self.shape: tuple[int] = (len(self.action_list),)
        self._action_to_index: dict[Action: int] = {action_: i for i, action_ in enumerate(self.action_list)}

    def _build_action_list(self):
        pass

    def _four_actions(self):
        self.action_list = [
            # left
            Action(move=common.XY(-1, 0)),
            # right
            Action(move=common.XY(+1, 0)),
            # up
            Action(move=common.XY(0, +1)),
            # down
            Action(move=common.XY(0, -1))
        ]

    def _four_friendly_actions(self):
        self.action_list = [
            # right
            Action(move=common.XY(+1, 0)),
            # up
            Action(move=common.XY(0, +1)),
            # left
            Action(move=common.XY(-1, 0)),
            # down
            Action(move=common.XY(0, -1))
        ]

    def _kings_moves(self, include_center: bool = False):
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                include: bool = True
                if x == 0 and y == 0:
                    include = include_center
                if include:
                    self.action_list.append(Action(move=common.XY(x, y)))

    def get_action_from_index(self, index: tuple[int]) -> Action:
        return self.action_list[index[0]]

    def get_index_from_action(self, action_: Action) -> tuple[int]:
        index = (self._action_to_index[action_], )
        return index


actions = Actions()
