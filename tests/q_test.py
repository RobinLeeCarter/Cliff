from __future__ import annotations

import common
import environment
import environments
from algorithm.value_function import state_action_function


def q_test() -> bool:
    environment_ = environments.Cliff()
    q = state_action_function.StateActionFunction(environment_, initial_q_value=-7.0)

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
