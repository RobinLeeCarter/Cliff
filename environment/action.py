from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import common


@dataclass(frozen=True)
class Action:
    move: common.XY

    @property
    def index(self) -> tuple[int]:
        return Actions.singleton.get_index_from_action(self)


class Actions:
    singleton: Optional[Actions] = None

    def __init__(self, actions_list: common.ActionsList):
        Actions.singleton = self
        self.action_list: list[Action] = self._build_action_list(actions_list)
        self.shape: tuple[int] = (len(self.action_list),)
        self._action_to_index: dict[Action: int] = {action_: i for i, action_ in enumerate(self.action_list)}

    def get_action_from_index(self, index: tuple[int]) -> Action:
        return self.action_list[index[0]]

    def get_index_from_action(self, action_: Action) -> tuple[int]:
        index = (self._action_to_index[action_], )
        return index

    def _build_action_list(self, actions_list: common.ActionsList) -> list[Action]:
        al = common.ActionsList
        if actions_list == al.STATIONARY_MOVE_ONLY:
            return self._stationary_move_only()
        elif actions_list == al.FOUR_MOVES:
            return self._four_moves()
        elif actions_list == al.FOUR_CLIFF_FRIENDLY_MOVES:
            return self._four_cliff_friendly_moves()
        elif actions_list == al.KINGS_MOVES:
            return self._kings_moves()
        elif actions_list == al.KINGS_MOVES_PLUS_NO_MOVE:
            return self._kings_moves(include_center=True)
        else:
            raise NotImplementedError

    # common moves that could be reused
    def _stationary_move_only(self) -> list[Action]:
        return [
            # stationary
            Action(move=common.XY(0, 0)),
        ]

    def _four_moves(self) -> list[Action]:
        return [
            # left
            Action(move=common.XY(-1, 0)),
            # right
            Action(move=common.XY(+1, 0)),
            # up
            Action(move=common.XY(0, +1)),
            # down
            Action(move=common.XY(0, -1))
        ]

    def _four_cliff_friendly_moves(self) -> list[Action]:
        return [
            # right
            Action(move=common.XY(+1, 0)),
            # up
            Action(move=common.XY(0, +1)),
            # left
            Action(move=common.XY(-1, 0)),
            # down
            Action(move=common.XY(0, -1))
        ]

    def _kings_moves(self, include_center: bool = False) -> list[Action]:
        action_list: list[Action] = []
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                include: bool = True
                if x == 0 and y == 0:
                    include = include_center
                if include:
                    action_list.append(Action(move=common.XY(x, y)))
        return action_list
