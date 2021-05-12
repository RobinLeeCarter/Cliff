from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment
    from mdp.model.agent.agent import Agent
from mdp import common
from mdp.model.algorithm.policy_evaluation.policy_evaluation_dp_v_np_deterministic import PolicyEvaluationDpVNp
from mdp.model.algorithm.policy_improvement.policy_improvement_dp_v_np_deterministic import PolicyImprovementDpVNp


class PolicyIterationDpVNp(PolicyEvaluationDpVNp, PolicyImprovementDpVNp):
    def __init__(self,
                 environment_: Environment,
                 agent_: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment_, agent_, algorithm_parameters, policy_parameters)
        self._algorithm_type = common.AlgorithmType.POLICY_ITERATION_DP_V
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name} θ={self._theta}"

    # @profile
    def run(self):
        if self._verbose:
            print(f"Starting Policy Iteration ...")

        iteration: int = 1
        policy_stable: bool = False
        cont: bool = True
        if self._step_callback:
            cont = self._step_callback()
        while cont and not policy_stable and iteration < self._iteration_timeout:
            if self._verbose:
                print(f"Policy Iteration. Iteration = {iteration}")
            self._policy_evaluation()
            policy_stable = self._policy_improvement()
            if self._step_callback:
                cont = self._step_callback()
            iteration += 1

        if iteration == self._iteration_timeout:
            print(f"Warning: Timed out at {iteration} iterations")
        else:
            # noinspection PyTypeChecker
            # policy: Deterministic = self._agent.policy
            # policy_vector: np.ndarray = policy.get_policy_vector()
            # policy.set_policy_vector(policy_vector, update_dict=True)

            if self._verbose:
                print(f"Policy Iteration completed ...")

        # print(f"iterations: {iteration-1}")
