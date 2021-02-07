import constants
import environment
import agent
from algorithm import abstract


class QLearning(abstract.EpisodicOnline):
    name: str = "Q-learning"

    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 alpha: float = 0.5,
                 verbose: bool = False
                 ):
        super().__init__(environment_, agent_, verbose)
        self.title = f"{QLearning.name} α={alpha}"
        self._alpha = alpha

    def _do_training_step(self):
        self.agent.choose_action()
        self.agent.take_action()

        prev_state = self.agent.prev_state
        prev_action = self.agent.prev_action
        reward = self.agent.reward
        state = self.agent.state

        q_max_over_a = self._Q.max_over_actions(state)
        delta = reward \
            + constants.GAMMA * q_max_over_a \
            - self._Q[prev_state, prev_action]
        self._Q[prev_state, prev_action] += self._alpha * delta
        # update policy to be in-line with Q
        self.agent.policy[prev_state] = self._Q.argmax_over_actions(prev_state)
