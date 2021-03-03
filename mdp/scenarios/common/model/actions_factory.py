from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.environment import action

from mdp import common
from mdp.scenarios.common.model import action_move  # , action_acceleration


class ActionsFactory:
    def __init__(self, actions_list: common.ActionsList):
        self.actions: list[action.Action] = self._build_action_list(actions_list)

    def _build_action_list(self, actions_list: common.ActionsList) -> list[action.Action]:
        al = common.ActionsList
        if actions_list == al.NO_ACTIONS:
            return self._no_actions()
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
    def _no_actions(self) -> list[action.Action]:
        return []

    def _four_moves(self) -> list[action_move.ActionMove]:
        return [
            # left
            action_move.ActionMove(move=common.XY(-1, 0)),
            # right
            action_move.ActionMove(move=common.XY(+1, 0)),
            # up
            action_move.ActionMove(move=common.XY(0, +1)),
            # down
            action_move.ActionMove(move=common.XY(0, -1))
        ]

    def _four_cliff_friendly_moves(self) -> list[action_move.ActionMove]:
        return [
            # right
            action_move.ActionMove(move=common.XY(+1, 0)),
            # up
            action_move.ActionMove(move=common.XY(0, +1)),
            # left
            action_move.ActionMove(move=common.XY(-1, 0)),
            # down
            action_move.ActionMove(move=common.XY(0, -1))
        ]

    def _kings_moves(self, include_center: bool = False) -> list[action_move.ActionMove]:
        action_list: list[action_move.ActionMove] = []
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                include: bool = True
                if x == 0 and y == 0:
                    include = include_center
                if include:
                    action_list.append(action_move.ActionMove(move=common.XY(x, y)))
        return action_list
