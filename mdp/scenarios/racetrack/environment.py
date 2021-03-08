from __future__ import annotations
from typing import Optional

from mdp import common
from mdp.model import environment
from mdp.scenarios.racetrack import action, grid_world, environment_parameters, state


class Environment(environment.Environment):
    def __init__(self, environment_parameters_: environment_parameters.EnvironmentParameters):
        grid_world_ = grid_world.GridWorld(environment_parameters_)
        super().__init__(environment_parameters_, grid_world_)

        # downcast states and actions so properties can be used freely
        self.states: list[state.State] = self.states
        self.actions: list[action.Action] = self.actions
        self._state: state.State = self._state
        self._action: action.Action = self._action
        self.grid_world: grid_world.GridWorld = self.grid_world

        self._reward: float = 0.0
        self._next_state: Optional[state.State] = None
        self._extra_reward_for_failure: float = environment_parameters_.extra_reward_for_failure

        # velocity
        self._min_vx: int = environment_parameters_.min_velocity
        self._max_vx: int = environment_parameters_.max_velocity
        self._min_vy: int = environment_parameters_.min_velocity
        self._max_vy: int = environment_parameters_.max_velocity

        # acceleration
        self._min_ax: int = environment_parameters_.min_acceleration
        self._max_ax: int = environment_parameters_.max_acceleration
        self._min_ay: int = environment_parameters_.min_acceleration
        self._max_ay: int = environment_parameters_.max_acceleration

    # region Sets
    def _build_states(self):
        """set S"""
        for x in range(self.grid_world.max_x+1):
            for y in range(self.grid_world.max_y+1):
                position: common.XY = common.XY(x=x, y=y)
                is_terminal: bool = self.grid_world.is_at_goal(position)
                for vx in range(self._min_vx, self._max_vx + 1):
                    for vy in range(self._min_vy, self._max_vy + 1):
                        new_state: state.State = state.State(
                            is_terminal=is_terminal,
                            position=position,
                            velocity=common.XY(x=vx, y=vy)
                        )
                        self.states.append(new_state)

    def _build_actions(self):
        for ax in range(self._min_ax, self._max_ax + 1):
            for ay in range(self._min_ay, self._max_ay + 1):
                new_action: action.Action = action.Action(
                    acceleration=common.XY(x=ax, y=ay)
                )
                self.actions.append(new_action)

    def is_action_compatible_with_state(self, state_: state.State, action_: action.Action):
        new_vx = state_.velocity.x + action_.acceleration.x
        new_vy = state_.velocity.y + action_.acceleration.y
        if self._min_vx <= new_vx <= self._max_vx and \
            self._min_vy <= new_vy <= self._max_vy and \
                not (new_vx == 0 and new_vy == 0):
            return True
        else:
            return False
    # endregion

    # region Operation
    def _get_a_start_state(self) -> state.State:
        position: common.XY = self.grid_world.get_a_start_position()
        return state.State(is_terminal=False, position=position, velocity=common.XY(x=0, y=0))

    def _apply_action(self):
        if not self.is_action_compatible_with_state(self._state, self._action):
            raise Exception(f"apply_action_to_state state {self._state} incompatible with action {self._action}")

        # apply grid world rules (eg. edges, wind)
        acceleration: Optional[common.XY] = None
        if self._action:
            acceleration = self._action.acceleration
        new_position, new_velocity = self.grid_world.change_request(
            position=self._state.position,
            velocity=self._state.velocity,
            acceleration=acceleration
            )

        self._square = self.grid_world.get_square(new_position)
        if self._square == common.Square.END:
            # success
            self._reward = 0.0
            self._next_state = state.State(
                position=new_position,
                velocity=new_velocity,
                is_terminal=True
            )
            if self.verbose:
                print(f"Past finish line at {new_position}")
        elif self._square == common.Square.CLIFF:
            # failure, move back to start line
            # self.pre_reset_state = state.State(x, y, vx, vy, is_reset=True)
            self._reward = -1.0 + self._extra_reward_for_failure
            self._next_state = self._get_a_start_state()
            if self.verbose:
                print(f"Grass at {new_position}")
        else:
            # TRACK or START so continue
            self._reward = -1.0
            self._next_state = state.State(
                position=new_position,
                velocity=new_velocity,
                is_terminal=False
            )

    def _get_response(self) -> environment.Response:
        return environment.Response(
            reward=self._reward,
            state=self._next_state
        )

    def output_mode(self):
        self.grid_world.skid_probability = 0.0
    # endregion
