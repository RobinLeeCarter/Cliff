from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model import environment, agent
    from mdp.model.policy.policy import Policy
from mdp import common
from mdp.model.algorithm import abstract


class PolicyImprovementDpQ(abstract.DynamicProgrammingQ):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment_, agent_, algorithm_parameters, policy_parameters)
        self._algorithm_type = common.AlgorithmType.POLICY_IMPROVEMENT_DP_Q
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name}"

    def run(self):
        do_call_back: bool = bool(self._step_callback)
        policy_stable: bool = False
        while not policy_stable:
            policy_stable = self._policy_improvement(do_call_back)

    def _policy_improvement(self, do_call_back: bool = False) -> bool:
        policy_: Policy = self._agent.target_policy
        # assert isinstance(policy_, policy.Deterministic)
        # policy_: policy.Deterministic

        if self._verbose:
            print(f"Starting Policy Improvement ...")

        policy_stable: bool = True
        for state in self._environment.states:
            old_action: environment.Action = policy_[state]
            best_action: environment.Action = self.Q.argmax_over_actions(state)
            policy_[state] = best_action
            if old_action != best_action:
                policy_stable = False

        if self._verbose:
            print(f"Policy Improvement completed. policy_stable = {policy_stable}")
        if do_call_back:
            self._step_callback()

        return policy_stable
