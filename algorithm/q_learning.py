import constants
import environment
import agent
from algorithm import episodic_algorithm


class QLearning(episodic_algorithm.EpisodicAlgorithm):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 alpha: float = 0.5,
                 verbose: bool = False
                 ):
        title = f"Q-learning alpha={alpha}"
        super().__init__(environment_, agent_, title, verbose)
        self._alpha = alpha

    def _do_training_step(self):
        self.agent.choose_action()
        self.agent.take_action()
        sarsa = self.agent.get_sarsa()
        q_max_over_a = self._Q.max_over_actions(sarsa.next_state)
        delta = sarsa.reward + \
            constants.GAMMA * q_max_over_a - \
            self._Q[sarsa.state, sarsa.action]
        self._Q[sarsa.state, sarsa.action] += self._alpha * delta
        self.agent.policy[sarsa.state] = self._Q.argmax_over_actions(sarsa.state)
