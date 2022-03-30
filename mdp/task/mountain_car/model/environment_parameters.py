from __future__ import annotations
from dataclasses import dataclass, field

from mdp import common
from mdp.task.mountain_car.model.action import Action


def actions_default_factory():
    actions = [Action(acceleration=-1.0),
               Action(acceleration=0.0),
               Action(acceleration=1.0)]
    return actions


@dataclass
class EnvironmentParameters(common.EnvironmentParameters):
    environment_type: common.EnvironmentType = common.EnvironmentType.MOUNTAIN_CAR
    actions_always_compatible: bool = True
    actions: list[Action] = field(default_factory=actions_default_factory)
    position_min: float = -1.2
    position_max: float = +0.5
    velocity_min: float = -0.07
    velocity_max: float = +0.07
