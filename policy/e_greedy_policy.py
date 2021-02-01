import numpy as np

import environment
import policy


class EGreedyPolicy(policy.RandomPolicy):
    def __init__(self, environment_: environment.Environment, rng: np.random.Generator,
                 greedy_policy: policy.DeterministicPolicy, epsilon: float = 0.1):
        super().__init__(environment_, rng)
        self.greedy_policy: policy.DeterministicPolicy = greedy_policy
        self.epsilon = epsilon

    def __getitem__(self, state: environment.State) -> environment.Action:
        if self.rng.uniform() > self.epsilon:
            return self.greedy_policy[state]
        else:
            return policy.RandomPolicy.__getitem__(self, state)
            # self.set_possible_actions(state_)
            # return self.rng.choice(self.possible_actions)

    def __setitem__(self, state: environment.State, action: environment.Action):
        self.greedy_policy[state] = action

    def get_probability(self, state_: environment.State, action_: environment.Action) -> float:
        self.set_possible_actions(state_)
        non_greedy_p = self.epsilon * (1.0 / len(self.possible_actions))
        if action_ == self.greedy_policy[state_]:
            greedy_p = (1 - self.epsilon) + non_greedy_p
            return greedy_p
        else:
            return non_greedy_p
