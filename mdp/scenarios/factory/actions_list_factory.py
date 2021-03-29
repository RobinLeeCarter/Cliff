from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.environment.action import Action

from mdp import common
from mdp.scenarios.position_move import action as pm


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


def _four_moves() -> list[pm.Action]:
    return [
        # left
        pm.Action(move=common.XY(-1, 0)),
        # right
        pm.Action(move=common.XY(+1, 0)),
        # up
        pm.Action(move=common.XY(0, +1)),
        # down
        pm.Action(move=common.XY(0, -1))
    ]


def _four_cliff_friendly_moves() -> list[pm.Action]:
    return [
        # right
        pm.Action(move=common.XY(+1, 0)),
        # up
        pm.Action(move=common.XY(0, +1)),
        # left
        pm.Action(move=common.XY(-1, 0)),
        # down
        pm.Action(move=common.XY(0, -1))
    ]


def _kings_moves(include_center: bool = False) -> list[pm.Action]:
    action_list: list[pm.Action] = []
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            include: bool = True
            if x == 0 and y == 0:
                include = include_center
            if include:
                action_list.append(pm.Action(move=common.XY(x, y)))
    return action_list
