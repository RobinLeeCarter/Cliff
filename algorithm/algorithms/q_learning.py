import constants
import environment
import agent
from algorithm.algorithms import episodic_algorithm


class QLearning(episodic_algorithm.EpisodicAlgorithm):
    name: str = "Q-learning"

    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 alpha: float = 0.5,
                 verbose: bool = False
                 ):
        title = f"{QLearning.name} α={alpha}"
        super().__init__(environment_, agent_, title, verbose)
        self._alpha = alpha

    def _do_training_step(self):
        self.agent.choose_action()
        self.agent.take_action()
        sarsa = self.agent.get_sarsa()
        q_max_over_a = self._Q.max_over_actions(sarsa.state)
        delta = sarsa.reward \
            + constants.GAMMA * q_max_over_a \
            - self._Q[sarsa.prev_state, sarsa.prev_action]
        self._Q[sarsa.prev_state, sarsa.prev_action] += self._alpha * delta
        self.agent.policy[sarsa.prev_state] = self._Q.argmax_over_actions(sarsa.prev_state)
