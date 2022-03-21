from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Optional, Generator

if TYPE_CHECKING:
    pass
from mdp.model.tabular.environment.tabular_dynamics import state_distribution


class Dynamics:
    """p(s',r|s,a)"""
    def __init__(self):
        self._state_distributions: dict[tuple[state.BaseState, action.BaseAction], state_distribution.StateDistribution] = {}
        self._state: Optional[state.BaseState] = None
        self._action: Optional[action.BaseAction] = None
        self._state_distribution: Optional[state_distribution.StateDistribution] = None

    def add(self, state_: state.BaseState, action_: action.BaseAction,
            new_state: state.BaseState, reward: float, probability: float):
        """add s, a, s', r, p(s',r|s,a)"""
        state_action: tuple[state.BaseState, action.BaseAction] = (state_, action_)
        state_distribution_ = self._state_distributions.get(state_action)
        if state_distribution_:
            state_distribution_.add(new_state, reward, probability)
        else:
            state_distribution_ = state_distribution.StateDistribution()
            state_distribution_.add(new_state, reward, probability)
            self._state_distributions[state_action] = state_distribution_

    def set_state_action(self, state_: state.BaseState, action_: action.BaseAction):
        """set state and action before using the functions below"""
        self._state: state.BaseState = state_
        self._action: action.BaseAction = action_
        state_action: tuple[state.BaseState, action.BaseAction] = (state_, action_)
        self._state_distribution = self._state_distributions.get(state_action)

    @property
    def expected_reward(self) -> Optional[float]:
        if self._state_distribution:
            return self._state_distribution.expected_reward
        else:
            return None

    def next_states(self) -> Generator[(state.BaseState, float, float), None, None]:
        """iterator for tuple(s', E[r|s,a,s'], p(s'|s,a))"""
        yield from self._state_distribution.next_states()

    def states_and_rewards(self) -> Generator[(state.BaseState, float, float), None, None]:
        """iterator for tuple(s', r, p(s',r|s,a))"""
        yield from self._state_distribution.states_and_rewards()

    def draw(self) -> (state.BaseState, float):
        """get (s',r)"""
        return self._state_distribution.draw()
