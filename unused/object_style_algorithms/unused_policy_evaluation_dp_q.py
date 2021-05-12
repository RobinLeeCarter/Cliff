from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment
    from mdp.model.agent.agent import Agent
from mdp import common
from mdp.model.algorithm.abstract.dynamic_programming_q import DynamicProgrammingQ


class PolicyEvaluationDpQ(DynamicProgrammingQ):
    def __init__(self,
                 environment_: Environment,
                 agent_: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment_, agent_, algorithm_parameters, policy_parameters)
        self._algorithm_type = common.AlgorithmType.POLICY_EVALUATION_DP_Q
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name} θ={self._theta}"

    def run(self):
        do_call_back: bool = bool(self._step_callback)
        self._policy_evaluation(do_call_back)

    def _policy_evaluation(self, do_call_back: bool = False):
        iteration: int = 1
        delta: float = float('inf')
        cont: bool = True

        if self._verbose:
            print(f"Starting Policy Evaluation ...")

        while cont and delta >= self._theta and iteration < self._iteration_timeout:
            delta = 0.0
            for state in self._environment.states:
                for action in self._environment.actions_for_state[state]:
                    q = self.Q[state, action]
                    new_q: float = self._get_expected_return(state, action)
                    self.Q[state, action] = new_q
                    delta = max(delta, abs(new_q - q))
            if self._verbose:
                print(f"iteration = {iteration}\tdelta={delta:.2f}")
            if do_call_back:
                cont = self._step_callback()
            iteration += 1

        if iteration == self._iteration_timeout:
            print(f"Warning: Timed out at {iteration} iterations")
        else:
            if self._verbose:
                print(f"Policy Evaluation completed. delta={delta:.2f}")

    def _get_expected_return(self, state: State, action: Action) -> float:
        # expected_return: Sum_over_s'_r(
        #   p(s',r|s,a).(r + γ.Sum_over_a'( π(a'|s').Q(s',a') ) )
        # )
        # RHS: Sum_over_s'_r( p(s',r|s,a).r)
        expected_reward: float = self._dynamics.get_expected_reward(state, action)

        # Sum_over_s'( p(s'|s,a) . Sum_over_a'( π(a'|s').Q(s',a') )
        next_state_expected_return: float = 0.0
        # s', p(s'|s,a)
        for next_state, probability in self._dynamics.get_state_transition_distribution(state, action).items():
            # Sum_over_a'( π(a'|s').Q(s',a') )
            next_state_return: float = 0

            for next_action in self._environment.actions_for_state[next_state]:
                # π(a'|s')
                policy_probability = self._agent.policy.get_probability(next_state, next_action)
                if policy_probability > 0:
                    # π(a'|s').Q(s',a')
                    next_state_return += policy_probability * self.Q[next_state, next_action]

            next_state_expected_return += probability * next_state_return

        expected_return = expected_reward + self._agent.gamma * next_state_expected_return

        return expected_return
