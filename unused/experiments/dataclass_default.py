from __future__ import annotations

from dataclasses import dataclass, InitVar
from typing import Optional

from mdp.common import utils


@dataclass()
class XY:
    x: Optional[int]
    y: Optional[int]
    default: InitVar[Optional[XY]] = None
    no_default: InitVar[bool] = False

    def __post_init__(self, default, no_default):
        # Push comparison values or default values into most settings attributes if currently =None
        if not no_default:
            if default is None:
                default = default_xy
            utils.set_none_to_default(self, default)

        # if default is not None:
        #     utils.set_none_to_default(self, default)
        # else:


default_xy: XY = XY(
    x=5,
    y=3,
    no_default=True
)

print(default_xy)

l1_xy: XY = XY(
    x=None,
    y=None,
)

print(l1_xy)

l2_xy: XY = XY(
    x=4,
    y=None,
    default=l1_xy
)

print(l2_xy)
