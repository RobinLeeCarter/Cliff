from __future__ import annotations
from typing import TYPE_CHECKING
import abc

if TYPE_CHECKING:
    from mdp.model import environment, agent
    from mdp import common
from mdp.model.algorithm.abstract import dynamic_programming


class DynamicProgrammingQ(dynamic_programming.DynamicProgramming, abc.ABC):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment_, agent_, algorithm_parameters, policy_parameters)
        self._create_q()

    def _get_expected_return(self, state: environment.State, action: environment.Action) -> float:
        # expected_return: Sum_over_s'_r(
        #   p(s',r|s,a).(r + γ.Sum_over_a'( π(a'|s').Q(s',a') ) )
        # )
        # RHS: Sum_over_s'_r( p(s',r|s,a).r)
        expected_return: float = self._dynamics.get_expected_reward(state, action)

        # Sum_over_s'( p(s'|s,a) . Sum_over_a'( π(a'|s').Q(s',a') )
        next_state_expected_return: float = 0.0
        # s', p(s'|s,a)
        for next_state, probability in self._dynamics.get_state_transition_distribution(state, action).items():
            # Sum_over_a'( π(a'|s').Q(s',a') )
            next_state_return: float = 0

            for next_action in self._environment.actions_for_state(next_state):
                # π(a'|s')
                policy_probability = self._agent.policy.get_probability(next_state, next_action)
                if policy_probability > 0:
                    # π(a'|s').Q(s',a')
                    next_state_return += policy_probability * self.Q[next_state, next_action]

            next_state_expected_return += probability * next_state_return

        expected_return += self._agent.gamma * next_state_expected_return

        return expected_return
