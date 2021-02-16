from __future__ import annotations

import common
from model import environment, scenarios
from model.algorithm.value_function import state_action_function


def q_test() -> bool:
    environment_parameters = common.EnvironmentParameters(
        environment_type=common.EnvironmentType.CLIFF,
        actions_list=common.ActionsList.FOUR_MOVES
    )
    environment_ = scenarios.environment_factory(environment_parameters)
    q = state_action_function.StateActionFunction(environment_, initial_q_value=-7.0)
    print(environment_.actions_shape)
    print(environment_.states_shape)

    state_ = environment.State(common.XY(x=4, y=2))
    print(f"state_.index {state_.index}")
    action_ = environment.Action(common.XY(x=1, y=0))
    print(f"action_.index {action_.index}")

    print(q[state_, action_])
    q[state_, action_] = 2.0
    q[state_, action_] += 0.5
    print(q[state_, action_])

    # noinspection PyProtectedMember
    print(f"q {q._values}")

    return True


if __name__ == '__main__':
    if q_test():
        print("Passed")
