from __future__ import annotations

from mdp import common
from mdp.model import environment
from mdp.scenarios.jacks import state, action, environment_parameters   # grid_world


class Environment(environment.Environment):
    def __init__(self, environment_parameters_: environment_parameters.EnvironmentParameters):
        # grid_world_ = grid_world.GridWorld(environment_parameters_)
        super().__init__(environment_parameters_, grid_world_=None)
        # super().__init__(environment_parameters_, grid_world_)
        self.dynamics = environment.Dynamics()

        # downcast states and actions so properties can be used freely
        self.states: list[state.State] = self.states
        self.actions: list[action.Action] = self.actions
        self._state: state.State = self._state
        self._action: action.Action = self._action

        self._max_cars: int = environment_parameters_.max_cars
        self._max_transfers: int = environment_parameters_.max_transfers

    # region Sets
    def _build_states(self):
        """set S"""
        for cars1 in range(self._max_cars+1):
            for cars2 in range(self._max_cars+1):
                new_state: state.State = state.State(
                    cars_cob_1=cars1,
                    cars_cob_2=cars2,
                    is_terminal=False,
                )
                self.states.append(new_state)

    def _build_actions(self):
        for cars in range(-self._max_transfers, self._max_transfers+1):
            new_action: action.Action = action.Action(
                transfer_1_to_2=cars
            )
            self.actions.append(new_action)

    def is_action_compatible_with_state(self, state_: state.State, action_: action.Action):
        new_cars_1 = state_.cars_cob_1 - action_.transfer_1_to_2
        new_cars_2 = state_.cars_cob_2 + action_.transfer_1_to_2
        if 0 <= new_cars_1 <= self._max_cars and \
                0 <= new_cars_2 <= self._max_cars:
            return True
        else:
            return False
    # endregion

    # region Dynamics
    def _build_dynamics(self):
        for state_ in self.states:
            for action_ in self.actions_for_state(state_):
                self._add_dynamics(state_, action_)

    def _add_dynamics(self, state_: state.State, action_: action.Action):
        new_cars_1 = state_.cars_cob_1 - action_.transfer_1_to_2
        new_cars_2 = state_.cars_cob_2 + action_.transfer_1_to_2


    # endregion



    def _get_response(self) -> environment.Response:
        reward: float
        self._new_state: state.State
        if self._new_state.position == common.XY(x=self.grid_world.max_x, y=0):
            reward = 1.0
        else:
            reward = 0.0

        return environment.Response(
            reward=reward,
            state=self._new_state
        )

    def get_optimum(self, state_: environment.State) -> float:
        self.grid_world: grid_world.GridWorld
        state_: state.State
        return self.grid_world.get_optimum(state_.position)
