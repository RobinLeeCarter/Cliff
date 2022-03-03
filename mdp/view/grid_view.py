from __future__ import annotations

from typing import Optional, TYPE_CHECKING, Callable
import abc
import sys

import pygame
import pygame.freetype
from matplotlib import cm, colors

if TYPE_CHECKING:
    from mdp.model.environment.tabular.grid_world import GridWorld
    from mdp.model.agent.tabular.episode import Episode
from mdp import common


class GridView(abc.ABC):
    def __init__(self, grid_view_parameters: common.GridViewParameters):
        self.grid_view_parameters: common.GridViewParameters = grid_view_parameters
        self._grid_world: Optional[GridWorld] = None

        self._max_x: Optional[int] = None
        self._max_y: Optional[int] = None

        self._screen_width: Optional[int] = None
        self._screen_height: Optional[int] = None
        self._cell_pixels: Optional[int] = None

        self._is_window_open: bool = False
        self._window_title: str = self.grid_view_parameters.window_title

        self._screen: Optional[pygame.Surface] = None
        self._background: Optional[pygame.Surface] = None
        self._grid_surface: Optional[pygame.Surface] = None

        self._background_color: Optional[pygame.Color] = None
        self._policy_color: Optional[pygame.Color] = None
        self._color_lookup: dict[int, pygame.Color] = {}
        self._agent_color: Optional[pygame.Color] = None
        self._agent_move_color: Optional[pygame.Color] = None
        self._prev_color: Optional[pygame.Color] = None
        self._prev_move_color: Optional[pygame.Color] = None

        self._policy_cmap: Optional[colors.Colormap] = None
        self._policy_color_normaliser: Optional[colors.Normalize] = None

        self._user_event: common.UserEvent = common.UserEvent.NONE

        self._t: int = 0
        self._episode: Optional[Episode] = None

        self._build_color_lookup()
        self._build_policy_color_map()

        # self._font: pygame.freetype.Font = pygame.freetype.Font(None, 12)
        self._font: pygame.freetype.Font = pygame.freetype.SysFont("Calibri", 12)

    @property
    def screen_size(self) -> tuple:
        return self._screen_width, self._screen_height

    # noinspection SpellCheckingInspection
    def _build_color_lookup(self):
        self._background_color = pygame.Color('grey10')
        self._policy_color = pygame.Color('pink')
        self._color_lookup = {
            common.Square.NORMAL: pygame.Color('grey66'),
            common.Square.CLIFF: pygame.Color('red2'),
            common.Square.START: pygame.Color('yellow2'),
            common.Square.END: pygame.Color('goldenrod2'),
        }
        self._agent_color = pygame.Color('deepskyblue2')
        self._agent_move_color = pygame.Color('forestgreen')
        self._prev_color = pygame.Color('grey76')
        self._prev_move_color = pygame.Color('forestgreen')

    def _build_policy_color_map(self):
        # noinspection PyUnresolvedReferences
        self._policy_cmap = cm.coolwarm
        self._policy_color_normaliser = colors.Normalize(vmin=0.0, vmax=1.0)

    def _get_policy_value_color(self, un_normalised: float) -> pygame.Color:
        normalised = self._policy_color_normaliser(un_normalised)
        rgba: common.RGBA = common.RGBA(*self._policy_cmap(normalised, bytes=True))
        color: pygame.Color = pygame.Color(rgba.as_tuple())
        return color

    def set_gridworld(self, grid_world: GridWorld):
        self._grid_world = grid_world
        self._max_x = self._grid_world.max_x
        self._max_y = self._grid_world.max_y
        self._load_gridworld()

    def _load_gridworld(self):
        self._set_sizes()
        self._grid_surface.fill(self._background_color)
        for x in range(self._max_x + 1):
            for y in range(self._max_y + 1):
                position: common.XY = common.XY(x, y)
                self._draw_square(self._grid_surface, position, draw_background=True)
        self._copy_grid_into_background()

    def _set_sizes(self):
        # size window for track and set cell_pixels
        rows, cols = self._grid_world.max_y + 1, self._grid_world.max_x + 1
        desired_screen_height = self.grid_view_parameters.screen_height
        desired_screen_width = self.grid_view_parameters.screen_width
        self._cell_pixels = int(min(desired_screen_height / rows, desired_screen_width / cols))
        self._screen_width = cols * self._cell_pixels
        self._screen_height = rows * self._cell_pixels

        self._background = pygame.Surface(size=self.screen_size)
        self._grid_surface = pygame.Surface(size=self.screen_size)

    def open_window(self):
        """open window if not already open
        safe to call repeatedly"""
        if not self._is_window_open:
            self._screen = pygame.display.set_mode(size=self.screen_size)
            pygame.display.set_caption(self._window_title)
            # self.background = pygame.Surface(size=self.screen_size).convert()
            self._background = self._background.convert()
            self._grid_surface = self._grid_surface.convert()
            pygame.key.set_repeat(500, 50)
            self._is_window_open = True
            # self.background.fill(self.background_color)

    def close_window(self):
        pygame.quit()
        self._is_window_open = False

    def _set_title(self, title: str):
        self._window_title = title
        pygame.display.set_caption(title)

    def display_and_wait(self):
        self.open_window()
        while self._user_event != common.UserEvent.QUIT:
            self._put_background_on_screen()
            # keys = pygame.key.get_pressed()
            # if keys[pygame.K_SPACE]:
            #     self.user_event = enums.enums.UserEvent.SPACE
            # else:
            self._wait_for_event_of_interest()
            # self._handle_event()

    def display_latest_step(self, episode_: Optional[Episode] = None):
        self.open_window()  # if not already
        self._copy_grid_into_background()
        if episode_:
            self._frame_on_background_latest(episode_)
        else:
            self._draw_frame_on_background()
        self._put_background_on_screen()
        self._wait_for_event_of_interest()
        if self._user_event == common.UserEvent.QUIT:
            self.close_window()
            sys.exit()

    def demonstrate(self, new_episode_request: Callable[[], Episode]):
        self.open_window()
        running_average = 0
        count = 0
        while True:
            count += 1
            episode: Episode = new_episode_request()
            print(f"max_t: {episode.max_t} \t total_return: {episode.total_return:.0f}")
            running_average += (1/count) * (episode.total_return - running_average)
            print(f"count: {count} \t running_average: {running_average:.1f}")
            user_event: common.UserEvent = self.display_episode(episode)
            if user_event == common.UserEvent.QUIT:
                break
        self.close_window()

    def display_episode(self, episode_: Episode) -> common.UserEvent:
        # print(episode_.trajectory)
        # print(f"len(self._episode.trajectory) = {len(episode_.trajectory)}")
        self._copy_grid_into_background()
        self._draw_frame_on_background()

        # self._put_background_on_screen()
        self._episode = episode_
        self._t = 0
        while self._user_event != common.UserEvent.QUIT and self._t <= self._episode.max_t:
            # need to pass through for terminal state to display penultimate state
            # keys = pygame.key.get_pressed()
            # if keys[pygame.K_SPACE]:
            #     self.user_event = enums.enums.UserEvent.SPACE
            # else:
            self._frame_on_background_for_t(episode_, self._t)
            self._put_background_on_screen()
            self._wait_for_event_of_interest()
            # self._handle_event(show_trail)
            self._t += 1
        return self._user_event

    def _frame_on_background_latest(self, episode_: Episode):
        """draw frame onto background for the latest state, action (& previous) from in-progress episode"""
        pass

    def _frame_on_background_for_t(self, episode_: Episode, t: int):
        """draw frame onto background for S(t), A(t) (& previous) from episode"""
        pass

    def _copy_grid_into_background(self):
        self._background.blit(source=self._grid_surface, dest=(0, 0))

    def _draw_v(self, surface: pygame.Surface, rect: pygame.Rect, output_square: common.OutputSquare):
        if output_square.v_value is not None:
            text: str = f"{output_square.v_value:.1f}"
            sub_rect = self._get_sub_rect(rect, move=common.XY(x=0, y=0))
            self._center_text(surface, sub_rect, text)

    def _draw_policy(self, surface: pygame.Surface, rect: pygame.Rect, output_square: common.OutputSquare):
        if output_square.policy_value is not None:
            text: str = f"{output_square.policy_value:.1f}"
            sub_rect = self._get_sub_rect(rect, move=common.XY(x=0, y=0))
            self._center_text(surface, sub_rect, text)

    def _draw_q(self, surface: pygame.Surface, rect: pygame.Rect, output_square: common.OutputSquare):
        for move_value in output_square.move_values.values():
            if move_value.q_value is not None:
                text: str = f"{move_value.q_value:.1f}"
                sub_rect = self._get_sub_rect(rect, move_value.move)
                if move_value.is_policy:
                    pygame.draw.rect(surface, self._policy_color, sub_rect)
                self._center_text(surface, sub_rect, text)

    def _draw_move(self, surface: pygame.Surface, rect: pygame.Rect, move: common.XY, move_color: pygame.Color):
        sub_rect = self._get_sub_rect(rect, move=move)
        width: int = 2
        pygame.draw.rect(surface, move_color, sub_rect, width)

    def _put_background_on_screen(self):
        self._screen.blit(source=self._background, dest=(0, 0))
        pygame.display.flip()

    def _wait_for_event_of_interest(self):
        self._user_event = common.UserEvent.NONE
        pygame.event.clear()
        while self._user_event == common.UserEvent.NONE:
            # replaced: for event in pygame.event.get():
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                self._user_event = common.UserEvent.QUIT
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self._user_event = common.UserEvent.SPACE

    def _draw_frame_on_background(self,
                                  agent_position: Optional[common.XY] = None,
                                  agent_move: Optional[common.XY] = None,
                                  prev_position: Optional[common.XY] = None,
                                  prev_move: Optional[common.XY] = None
                                  ):
        for x in range(self._max_x + 1):
            for y in range(self._max_y + 1):
                position: common.XY = common.XY(x, y)
                if position == agent_position:
                    self._draw_agent_on_background(agent_position, agent_move)
                elif position == prev_position:
                    self._draw_prev_on_background(prev_position, prev_move)
                else:
                    self._draw_square(surface=self._background,
                                      position=position
                                      )

    def _draw_agent_on_background(self,
                                  agent_position: Optional[common.XY] = None,
                                  agent_move: Optional[common.XY] = None,
                                  ):
        self._draw_square(surface=self._background,
                          position=agent_position,
                          position_color=self._agent_color,
                          draw_background=True,
                          move=agent_move,
                          move_color=self._agent_move_color,
                          draw_move=True
                          )

    def _draw_prev_on_background(self,
                                 prev_position: Optional[common.XY],
                                 prev_move: Optional[common.XY],
                                 ):
        self._draw_square(surface=self._background,
                          position=prev_position,
                          position_color=self._prev_color,
                          draw_background=True,
                          move=prev_move,
                          move_color=self._prev_move_color,
                          draw_move=True
                          )

    def _draw_square(self,
                     surface: pygame.Surface,
                     position: common.XY,
                     position_color: Optional[pygame.color] = None,
                     draw_background: bool = False,
                     move: Optional[common.XY] = None,
                     move_color: Optional[pygame.color] = None,
                     draw_move: bool = False,
                     draw_v: Optional[bool] = None,
                     draw_policy: Optional[bool] = None,
                     draw_q: Optional[bool] = None
                     ):
        if draw_v is None:
            draw_v = self.grid_view_parameters.show_v
        if draw_q is None:
            draw_q = self.grid_view_parameters.show_q
        if draw_policy is None:
            draw_policy = self.grid_view_parameters.show_policy

        # make rect
        row = self._max_y - position.y
        col = position.x

        left: int = col * self._cell_pixels
        top: int = row * self._cell_pixels
        width: int = self._cell_pixels - 1
        height: int = self._cell_pixels - 1

        # doesn't like named parameters
        rect: pygame.Rect = pygame.Rect(left, top, width, height)

        if draw_background:
            if not position_color:
                square: int = self._grid_world.get_square(position)
                position_color: pygame.Color = self._color_lookup[square]
            pygame.draw.rect(surface, position_color, rect)

        output_square: common.OutputSquare = self._grid_world.output_squares[row, col]
        if draw_v:
            self._draw_v(surface, rect, output_square)
        if draw_policy:
            self._draw_policy(surface, rect, output_square)
        if draw_q:
            self._draw_q(surface, rect, output_square)

        if draw_move and move:
            self._draw_move(surface, rect, move, move_color)

    def _get_sub_rect(self, rect: pygame.Rect, move: common.XY) -> pygame.Rect:
        sub_width: float = rect.width / 3.0
        sub_height: float = rect.height / 3.0
        sub_left: float = rect.left + (move.x + 1)*sub_width
        sub_top: float = rect.top + (1 - move.y)*sub_height
        sub_rect: pygame.Rect = pygame.Rect(sub_left, sub_top, sub_width, sub_height)
        return sub_rect

    def _center_text(self, surface: pygame.Surface, rect: pygame.Rect, text: str):
        bounds = self._font.get_rect(text)
        bounds.center = (0, 0)
        bounds.move_ip(rect.center)
        self._font.render_to(surface, bounds.topleft, text)
