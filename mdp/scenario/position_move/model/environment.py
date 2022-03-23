from __future__ import annotations
from abc import ABC
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.tabular.algorithm.tabular_algorithm import TabularAlgorithm
    from mdp.model.tabular.policy.tabular_policy import TabularPolicy

from mdp import common
from mdp.scenario.position_move.model.state import State
from mdp.scenario.position_move.model.action import Action
from mdp.scenario.position_move.model import actions_list_factory
from mdp.scenario.position_move.model.grid_world import GridWorld
from mdp.scenario.position_move.model.dynamics import Dynamics

from mdp.model.tabular.environment.tabular_environment import TabularEnvironment


class Environment(TabularEnvironment[State, Action], ABC):
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        super().__init__(environment_parameters)
        self.grid_world: Optional[GridWorld] = None
        self.dynamics: Optional[Dynamics] = None

    # region Sets
    def _build_states(self):
        """set S"""
        for x in range(self.grid_world.max_x+1):
            for y in range(self.grid_world.max_y+1):
                position = common.XY(x=x, y=y)
                is_terminal: bool = self.grid_world.is_at_goal(position)
                new_state: State = State(
                    position=position,
                    is_terminal=is_terminal,
                )
                self.states.append(new_state)

    def _build_actions(self):
        actions_list = self._environment_parameters.actions_list
        self.actions: list[Action] = actions_list_factory.actions_list_factory(actions_list)

    def update_grid_value_functions(self,
                                    algorithm: TabularAlgorithm
                                    ):
        policy: TabularPolicy = algorithm.target_policy.linked_policy
        for s, state in enumerate(self.states):
            if algorithm.V:
                self.grid_world.set_v_value(
                    position=state.position,
                    v_value=algorithm.V[s]
                )
            if algorithm.Q:
                is_terminal: bool = self.is_terminal[s]
                policy_a: int = policy[s]
                # for a in np.flatnonzero(self.s_a_compatibility[s]):
                for a, action in enumerate(self.actions):
                    if self.s_a_compatibility[s, a]:
                        is_policy: bool = (not is_terminal and policy_a == a)
                        self.grid_world.set_move_q_value(
                            position=state.position,
                            move=action.move,
                            q_value=algorithm.Q[s, a],
                            is_policy=is_policy
                        )
