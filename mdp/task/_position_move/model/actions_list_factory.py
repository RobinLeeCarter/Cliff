from __future__ import annotations

from mdp import common
from mdp.task._position_move.model.action import Action


def actions_list_factory(actions_list: common.ActionsList) -> list[Action]:
    al = common.ActionsList
    if actions_list == al.NO_ACTIONS:
        return _no_actions()
    elif actions_list == al.FOUR_MOVES:
        return _four_moves()
    elif actions_list == al.FOUR_CLIFF_FRIENDLY_MOVES:
        return _four_cliff_friendly_moves()
    elif actions_list == al.KINGS_MOVES:
        return _kings_moves()
    elif actions_list == al.KINGS_MOVES_PLUS_NO_MOVE:
        return _kings_moves(include_center=True)
    else:
        raise NotImplementedError


# common moves that could be reused
def _no_actions() -> list[Action]:
    return []


def _four_moves() -> list[Action]:
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


def _four_cliff_friendly_moves() -> list[Action]:
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


def _kings_moves(include_center: bool = False) -> list[Action]:
    action_list: list[Action] = []
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            include: bool = True
            if x == 0 and y == 0:
                include = include_center
            if include:
                action_list.append(Action(move=common.XY(x, y)))
    return action_list
