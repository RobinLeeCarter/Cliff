from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mdp.model import environment, agent, policy
from mdp import common
from mdp.model.algorithm import abstract


class PolicyImprovementDpV(abstract.DynamicProgrammingV):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment_, agent_, algorithm_parameters, policy_parameters)
        self._algorithm_type = common.AlgorithmType.POLICY_IMPROVEMENT_DP_V
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name}"

    def run(self):
        self._policy_improvement()

    def _policy_improvement(self) -> bool:
        policy_: policy.Policy = self._agent.target_policy
        assert isinstance(policy_, policy.Deterministic)
        policy_: policy.Deterministic

        if self._verbose:
            print(f"Starting Policy Improvement ...")

        policy_stable: bool = True
        for state in self._environment.states:
            old_action: environment.Action = policy_[state]
            best_action: Optional[environment.Action] = None
            best_expected_return: float = float('-inf')
            for action in self._environment.actions_for_state(state):
                expected_return = self._get_expected_return(state, action)
                if expected_return > best_expected_return:
                    best_action = action
                    best_expected_return = expected_return
            policy_[state] = best_action
            if old_action != best_action:
                policy_stable = False

        if self._verbose:
            print(f"Policy Improvement completed. policy_stable = {policy_stable}")

        return policy_stable
