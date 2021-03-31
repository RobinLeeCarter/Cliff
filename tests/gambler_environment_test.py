from __future__ import annotations

import random

from mdp import common
from unused import unused_environment_factory
from mdp.scenarios.gambler import comparisons  # action, state,
from mdp.scenarios.gambler.model.state import State
from mdp.scenarios.gambler.model.action import Action
from mdp.scenarios.gambler.model.response import Response
from mdp.scenarios.gambler.model.environment import Environment


def gambler_test() -> bool:
    comparison: common.Comparison = comparisons.gambler_value_iteration_v()
    environment = unused_environment_factory.environment_factory(comparison.environment_parameters)

    # print("States...")
    # for state in environment.states:
    #     print(f"{state}")
    # print()
    #
    # print("Actions...")
    # for action in environment.actions:
    #     print(f"{action}")
    # print()

    # print(f"total states = {len(environment.states)}")
    # print(f"total actions = {len(environment.actions)}")

    # environment.initialize_policy()

    # state = State(is_terminal=False, player_sum=15, usable_ace=True, dealers_card=10)
    # action = Action(hit=False)

    # random_round(environment)
    # random_round(environment)
    # random_round(environment)
    # random_round(environment)
    # random_round(environment)
    #
    # state = State(is_terminal=False, capital=25)
    # action = Action(stake=25)
    # print(state, action)
    # response: Response = environment.from_state_perform_action(state, action)
    # print(response)
    # print()
    #
    # state = State(is_terminal=False, capital=25)
    # action = Action(stake=25)
    # print(state, action)
    # response: Response = environment.from_state_perform_action(state, action)
    # print(response)
    # print()
    #
    # state = State(is_terminal=False, capital=50)
    # action = Action(stake=50)
    # print(state, action)
    # response: Response = environment.from_state_perform_action(state, action)
    # print(response)
    # print()

    state = State(is_terminal=False, capital=50)
    action = Action(stake=50)
    print(state, action)
    expected_reward = environment.dynamics.get_expected_reward(state, action)
    distribution = environment.dynamics.get_state_transition_distribution(state, action)
    print(expected_reward)
    print(distribution)
    # response: Response = environment.from_state_perform_action(state, action)
    # print(response)
    # print()

    # print(state, action)
    # response: Response = environment.from_state_perform_action(state, action)
    # print(response)
    # print()
    #
    # print(state, action)
    # response: Response = environment.from_state_perform_action(state, action)
    # print(response)
    # print()
    #
    # print(state, action)
    # response: Response = environment.from_state_perform_action(state, action)
    # print(response)
    # print()

    # observation_ = environment.from_state_perform_action(state_, action_)
    # print(state_, action_)
    # print(observation_)
    #
    # state_ = state.State(is_terminal=False, position=common.XY(x=5, y=0))
    # action_ = action.Action(common.XY(x=1, y=0))
    # observation_ = environment.from_state_perform_action(state_, action_)
    # print(state_, action_)
    # print(observation_)
    #
    # state_ = state.State(is_terminal=False, position=common.XY(x=0, y=0))
    # action_ = action.Action(common.XY(x=-1, y=0))
    # observation_ = environment.from_state_perform_action(state_, action_)
    # print(state_, action_)
    # print(observation_)

    return True


def random_round(environment: Environment):
    response: Response = environment.start()
    state: State = response.state
    # noinspection PyProtectedMember
    max_stake = min(state.capital, environment._max_capital-state.capital)
    stake = random.choice(range(1, max_stake+1))
    action = Action(stake=stake)
    print(state, action)
    response: Response = environment.from_state_perform_action(state, action)
    print(response)
    print()


if __name__ == '__main__':
    if gambler_test():
        print("Passed")
