from __future__ import annotations


from mdp import common
from mdp.scenarios import scenario_factory
from mdp.scenarios.blackjack.model.state import State
from mdp.scenarios.blackjack.model.action import Action
from mdp.scenarios.blackjack.model.response import Response
from mdp.scenarios.blackjack.model.environment import Environment


def blackjack_test() -> bool:
    scenario = scenario_factory.scenario_factory(common.ComparisonType.BLACKJACK_EVALUATION_V)
    scenario.build()
    # noinspection PyProtectedMember
    environment: Environment = scenario._model.environment

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

    random_round(environment)
    random_round(environment)
    random_round(environment)
    random_round(environment)
    random_round(environment)

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
    response: Response = environment.start_state()
    state: State = response.state
    if state.player_sum >= 20:
        action = Action(hit=False)
    else:
        action = Action(hit=True)

    print(state, action)
    response: Response = environment.from_state_perform_action(state, action)
    print(response)
    print()


if __name__ == '__main__':
    if blackjack_test():
        print("Passed")
