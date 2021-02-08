from __future__ import annotations

import common
import environment
import data
from algorithm.common import state_action_function


def q_test() -> bool:
    environment_ = environment.Environment(grid_=data.CLIFF_GRID)
    q = state_action_function.StateActionFunction(environment_)

    state_ = environment.State(common.XY(x=4, y=2))
    action_ = environment.Action(common.XY(x=1, y=0))
    print(q[state_, action_])
    q[state_, action_] = 2.0
    q[state_, action_] += 0.5
    print(q[state_, action_])

    return True


if __name__ == '__main__':
    if q_test():
        print("Passed")
