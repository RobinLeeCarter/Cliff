from __future__ import annotations

from dataclasses import dataclass

import common


@dataclass(frozen=True)
class Action:
    # move before wind
    move: common.XY

    @property
    def index(self) -> tuple[int]:
        return Actions.get_index_from_action(self)


class Actions:
    action_list: list[Action]
    shape: tuple[int]
    _action_to_index: dict[Action: int]

    @staticmethod
    def init():
        # Actions.four_actions()
        Actions.kings_moves()
        # Actions.kings_moves(include_center=True)
        Actions.shape = (len(Actions.action_list),)
        Actions._action_to_index = {action_: i for i, action_ in enumerate(Actions.action_list)}

    @staticmethod
    def four_actions():
        Actions.action_list = [
            # left
            Action(move=common.XY(-1, 0)),
            # right
            Action(move=common.XY(1, 0)),
            # up
            Action(move=common.XY(0, +1)),
            # down
            Action(move=common.XY(0, -1))
        ]

    @staticmethod
    def kings_moves(include_center: bool = False):
        Actions.action_list = []
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                include: bool = True
                if x == 0 and y == 0:
                    include = include_center
                if include:
                    Actions.action_list.append(Action(move=common.XY(x, y)))

    @staticmethod
    def get_action_from_index(index: tuple[int]) -> Action:
        return Actions.action_list[index[0]]

    @staticmethod
    def get_index_from_action(action_: Action) -> tuple[int]:
        index = (Actions._action_to_index[action_], )
        return index


Actions.init()
# print(Actions.action_list)
