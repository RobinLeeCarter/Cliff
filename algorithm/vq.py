import constants
import environment
import agent
from algorithm import episodic_algorithm


class VQ(episodic_algorithm.EpisodicAlgorithm):
    name: str = "VQ"

    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 alpha: float = 0.5,
                 alpha_variable: bool = False,
                 verbose: bool = False
                 ):
        self.alpha_variable = alpha_variable
        if self.alpha_variable:
            title = f"{VQ.name} α=0.5 then α=0.1"
        else:
            title = f"{VQ.name} α={alpha}"
        super().__init__(environment_, agent_, title, verbose)
        self._alpha = alpha

    def parameter_changes(self, iteration: int):
        if self.alpha_variable:
            if iteration <= 50:
                self._alpha = 0.5
            else:
                self._alpha = 0.1

            # if iteration <= 20:
            #     self._alpha = 0.5
            # else:
            #     self._alpha = 10/iteration

    def _do_training_step(self):
        self.agent.choose_action()
        self.agent.take_action()

        prev_state = self.agent.prev_state
        prev_action = self.agent.action
        reward = self.agent.reward
        state = self.agent.state

        target = reward + constants.GAMMA * self._V[state]

        v_delta = target - self._V[prev_state]
        self._V[prev_state] += self._alpha * v_delta

        q_delta = target - self._Q[prev_state, prev_action]
        self._Q[prev_state, prev_action] += self._alpha * q_delta

        self.agent.policy[prev_state] = self._Q.argmax_over_actions(prev_state)
