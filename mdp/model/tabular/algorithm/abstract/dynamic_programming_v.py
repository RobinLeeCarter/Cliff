from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC

if TYPE_CHECKING:
    from mdp.model.tabular.agent.tabular_agent import TabularAgent
    from mdp import common
from mdp.model.tabular.algorithm.abstract.dynamic_programming import DynamicProgramming


class DynamicProgrammingV(DynamicProgramming, ABC):
    def __init__(self,
                 agent: TabularAgent,
                 algorithm_parameters: common.AlgorithmParameters,
                 name: str
                 ):
        super().__init__(agent, algorithm_parameters, name)
        self._create_v()

    # def _get_expected_return(self, state: State, action: Action) -> float:
    #     expected_return: float = self._dynamics.get_expected_reward(state, action)
    #     next_state_expected_return: float = 0.0
    #     for next_state, probability in self._dynamics.get_state_transition_distribution(state, action).items():
    #         next_state_expected_return += probability * self.V[next_state]
    #     expected_return += self._agent.gamma * next_state_expected_return
    #     return expected_return
    #
    # def _make_policy_greedy_wrt_v(self, round_first: bool):
    #     # round_first is recommmended to be True but will make it slightly slower
    #     for state in self._environment.states:
    #         if not state.is_terminal:
    #             action_values: dict[Action, float] = {}
    #             for action in self._environment.actions_for_state[state]:
    #                 value: float = self._get_expected_return(state, action)
    #                 if round_first:
    #                     value = round(value / self._theta, 0) * self._theta
    #                 action_values[action] = value
    #             # argmax https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
    #             best_action = max(action_values, key=action_values.get)
    #             self._agent.policy[state] = best_action

    # action_values: dict[environment.Action, float] = \
    #     {action: round(self._get_expected_return(state, action) / self._theta, 0)
    #      for action in self._environment.actions_for_state(state)}
    # if only_changes:
    #     max_value = max(action_values.values())
    #     current_action: environment.Action = self._agent.policy[state]
    #     current_value: float = action_values[current_action]
    #     if not math.isclose(current_value, max_value):
    #         self._agent.policy[state] = max(action_values, key=action_values.get)
    # else:
