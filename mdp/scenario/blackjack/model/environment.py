from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.tabular.algorithm.tabular_algorithm import TabularAlgorithm
    from mdp.model.tabular.policy.tabular_policy import TabularPolicy

from mdp import common
from mdp.scenario.blackjack.model.state import State
from mdp.scenario.blackjack.model.action import Action
from mdp.scenario.blackjack.comparison.environment_parameters import EnvironmentParameters
from mdp.scenario.blackjack.model.grid_world import GridWorld
from mdp.scenario.blackjack.model.dynamics import Dynamics

from mdp.model.tabular.environment.tabular_environment import TabularEnvironment


class Environment(TabularEnvironment[State, Action]):
    def __init__(self, environment_parameters: EnvironmentParameters):
        super().__init__(environment_parameters)
        self._environment_parameters: EnvironmentParameters = environment_parameters

        self.player_sum_min = 11
        self.player_sum_max = 21
        self.dealers_card_min = 1
        self.dealers_card_max = 10
        self.player_sums = [x for x in range(self.player_sum_min, self.player_sum_max + 1)]
        self.dealers_cards = [x for x in range(self.dealers_card_min, self.dealers_card_max + 1)]

        # dealer_card is x, player_sum is y : following the table in the book
        grid_shape = (len(self.player_sums), len(self.dealers_cards))
        self.grid_world: GridWorld = GridWorld(environment_parameters=environment_parameters, grid_shape=grid_shape)
        self.dynamics: Dynamics = Dynamics(environment=self, environment_parameters=environment_parameters)

    def _build_states(self):
        """set S"""
        # non-terminal states
        for player_sum in self.player_sums:
            for usable_ace in [False, True]:
                for dealers_card in self.dealers_cards:
                    new_state: State = State(
                        is_terminal=False,
                        player_sum=player_sum,
                        usable_ace=usable_ace,
                        dealers_card=dealers_card,
                    )
                    self.states.append(new_state)

        # terminal states
        for result in [-1, 0, +1]:
            new_state: State = State(
                is_terminal=True,
                result=result
            )
            self.states.append(new_state)

    def _build_actions(self):
        for hit in [False, True]:
            new_action: Action = Action(
                hit=hit
            )
            self.actions.append(new_action)

    def _is_action_compatible_with_state(self, state_: State, action_: Action):
        if state_.player_sum == 21 and action_.hit:
            return False
        else:
            return True

    def initialize_policy(self, policy: TabularPolicy):
        hit: bool

        policy.zero_state_action()
        for s, state in enumerate(self.states):
            # don't add an action to the policy for terminal states at all
            if not state.is_terminal:
                if state.player_sum >= 20:
                    hit = False
                else:
                    hit = True
                initial_action: Action = Action(hit)
                policy.set_action(s, initial_action)

    def update_grid_policy_ace(self, algorithm: TabularAlgorithm, usable_ace: bool):
        policy: TabularPolicy = algorithm.target_policy
        # policy_: policy.Deterministic
        for s, state in enumerate(self.states):
            if not state.is_terminal and state.usable_ace == usable_ace:
                # dealer_card is x, player_sum is y : following the table in the book
                x = state.dealers_card - self.dealers_card_min
                y = state.player_sum - self.player_sum_min
                position: common.XY = common.XY(x, y)
                action: Action = policy.get_action(s)   # type: ignore
                policy_value: int = int(action.hit)
                # print(position, transfer_1_to_2)
                self.grid_world.set_policy_value(
                    position=position,
                    policy_value=policy_value,
                )

                if algorithm.Q:
                    policy_a: int = policy[s]
                    is_terminal: bool = self.is_terminal[s]
                    for a, action in enumerate(self.actions):
                        if self.s_a_compatibility[s, a]:
                            is_policy: bool = (not is_terminal and policy_a == a)
                            if action.hit:
                                y = 1
                            else:
                                y = -1
                            move: common.XY = common.XY(0, y)
                            self.grid_world.set_move_q_value(
                                position=position,
                                move=move,
                                q_value=algorithm.Q[s, a],
                                is_policy=is_policy
                            )
