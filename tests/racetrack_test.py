from __future__ import annotations

from mdp import common
from mdp.scenarios.factory import environment_factory
from mdp.scenarios.racetrack import state, action, comparison, environment_parameters, grids


def racetrack_test() -> bool:
    comparison_ = comparison.Comparison(
        environment_parameters=environment_parameters.EnvironmentParameters(
            grid=grids.TRACK_1
        ),
        comparison_settings=common.Settings(
            runs=1,
            training_episodes=10_000,
            # display_every_step=True,
            dual_policy_relationship=common.DualPolicyRelationship.LINKED_POLICIES
        ),
        breakdown_parameters=common.BreakdownParameters(
            breakdown_type=common.BreakdownType.RETURN_BY_EPISODE
        ),
        settings_list=[
            # common.Settings(algorithm_parameters=common.AlgorithmParameters(
            #     algorithm_type=common.AlgorithmType.EXPECTED_SARSA,
            #     alpha=0.9
            # )),
            # common.Settings(algorithm_parameters=common.AlgorithmParameters(
            #     algorithm_type=common.AlgorithmType.VQ,
            #     alpha=0.2
            # )),
            # common.Settings(algorithm_parameters=common.AlgorithmParameters(
            #     algorithm_type=common.AlgorithmType.Q_LEARNING,
            #     alpha=0.5
            # )),
            common.Settings(algorithm_parameters=common.AlgorithmParameters(
                algorithm_type=common.AlgorithmType.OFF_POLICY_MC_CONTROL
            )),
        ],
        graph_values=common.GraphValues(
            moving_average_window_size=19,
            y_min=-200,
            y_max=0
        ),
        grid_view_parameters=common.GridViewParameters(
            grid_view_type=common.GridViewType.POSITION,
            show_demo=True,
            show_trail=True
        )
    )

    environment_ = environment_factory.environment_factory(comparison_.environment_parameters)

    for state_ in environment_.states:
        state_index = environment_.state_index[state_]
        print(f"{state_} \t index={state_index}")

    print()

    for action_ in environment_.actions:
        action_index = environment_.action_index[action_]
        print(f"{action_} \t index={action_index}")

    print()

    state_ = state.State(is_terminal=False, position=common.XY(x=4, y=0), velocity=common.XY(x=0, y=1))
    action_ = action.Action(acceleration=common.XY(x=1, y=0))
    response_ = environment_.from_state_perform_action(state_, action_)
    print(state_, action_)
    print(response_)

    state_ = state.State(is_terminal=False, position=common.XY(x=5, y=4), velocity=common.XY(x=1, y=0))
    action_ = action.Action(acceleration=common.XY(x=0, y=0))
    response_ = environment_.from_state_perform_action(state_, action_)
    print(state_, action_)
    print(response_)

    state_ = state.State(is_terminal=False, position=common.XY(x=0, y=0), velocity=common.XY(x=0, y=3))
    action_ = action.Action(common.XY(x=0, y=-1))
    response_ = environment_.from_state_perform_action(state_, action_)
    print(state_, action_)
    print(response_)

    return True


if __name__ == '__main__':
    if racetrack_test():
        print("Passed")
