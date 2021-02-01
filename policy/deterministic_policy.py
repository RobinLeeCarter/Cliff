import numpy as np

import environment
import policy


class DeterministicPolicy(policy.Policy):
    def __init__(self, environment_: environment.Environment):
        super().__init__(environment_)
        self._action_given_state: np.ndarray = np.empty(shape=self.environment.states_shape, dtype=environment.Action)

    def __getitem__(self, state: environment.State) -> environment.Action:
        return self._action_given_state[state.index]

    def __setitem__(self, state: environment.State, action: environment.Action):
        self._action_given_state[state.index] = action

    def get_probability(self, state_: environment.State, action_: environment.Action) -> float:
        if action_ == self[state_]:
            return 1.0
        else:
            return 0.0
