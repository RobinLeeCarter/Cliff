import numpy as np

import common
import environment
import data


rng: np.random.Generator = np.random.default_rng()


def environment_test() -> bool:
    environment_ = environment.Environment(grid_=data.GRID_1, rng=rng)

    for state_ in environment_.states():
        print(state_)

    print()

    for action_ in environment_.actions():
        print(action_)

    print()

    state_ = environment.State(common.XY(x=4, y=2))
    action_ = environment.Action(common.XY(x=1, y=0))
    response_ = environment_.from_state_perform_action(state_, action_)
    print(state_, action_)
    print(response_)

    return True


if __name__ == '__main__':
    if environment_test():
        print("Passed")
