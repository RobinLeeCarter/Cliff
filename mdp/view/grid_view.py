from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Callable

import pygame

if TYPE_CHECKING:
    from mdp.model import agent, environment
from mdp import common


class GridView:
    def __init__(self):
        self._grid_world: Optional[environment.GridWorld] = None

        self._max_x: Optional[int] = None
        self._max_y: Optional[int] = None

        self._screen_width: int = 1500
        self._screen_height: int = 1000
        self._title: str = "Not implemented"
        self._cell_pixels: int = 10
        self._screen: Optional[pygame.Surface] = None
        self._background: Optional[pygame.Surface] = None
        self._grid_surface: Optional[pygame.Surface] = None

        self._background_color: Optional[pygame.Color] = None
        self._color_lookup: dict[common.Square, pygame.Color] = {}

        self._user_event: common.UserEvent = common.UserEvent.NONE

        self._t: int = 0
        self._episode: Optional[agent.Episode] = None

        self._build_color_lookup()

    def set_grid_world(self, grid_world_: environment.GridWorld):
        self._grid_world = grid_world_
        self._max_x = self._grid_world.max_x
        self._max_y = self._grid_world.max_y
        self._load_gridworld()

    @property
    def screen_size(self) -> tuple:
        return self._screen_width, self._screen_height

    def open_window(self):
        self._screen = pygame.display.set_mode(size=self.screen_size)
        pygame.display.set_caption(self._title)
        # self.background = pygame.Surface(size=self.screen_size).convert()
        self._background = self._background.convert()
        self._grid_surface = self._grid_surface.convert()
        pygame.key.set_repeat(500, 50)
        # self.background.fill(self.background_color)

    def display_and_wait(self):
        while self._user_event != common.UserEvent.QUIT:
            self._put_background_on_screen()
            # keys = pygame.key.get_pressed()
            # if keys[pygame.K_SPACE]:
            #     self.user_event = enums.enums.UserEvent.SPACE
            # else:
            self._wait_for_event_of_interest()
            # self._handle_event()

    def demonstrate(self, new_episode_request: Callable[[], agent.Episode]):
        self.open_window()
        running_average = 0
        count = 0
        while True:
            count += 1
            episode: agent.Episode = new_episode_request()
            print(f"max_t: {episode.max_t} \t total_return: {episode.total_return:.0f}")
            running_average += (1/count) * (episode.total_return - running_average)
            print(f"count: {count} \t running_average: {running_average:.1f}")
            user_event: common.UserEvent = self.display_episode(episode, show_trail=False)
            if user_event == common.UserEvent.QUIT:
                break
        self.close_window()

    def display_episode(self, episode_: agent.Episode, show_trail: bool = True) -> common.UserEvent:
        # print(episode_.trajectory)
        # print(f"len(self._episode.trajectory) = {len(episode_.trajectory)}")
        self._copy_grid_into_background()
        self._put_background_on_screen()
        self._episode = episode_
        self._t = 0
        while self._user_event != common.UserEvent.QUIT and self._t <= self._episode.max_t:
            # need to pass through for terminal state to display penultimate state
            # keys = pygame.key.get_pressed()
            # if keys[pygame.K_SPACE]:
            #     self.user_event = enums.enums.UserEvent.SPACE
            # else:
            self._draw_agent(show_trail)
            self._wait_for_event_of_interest()
            # self._handle_event(show_trail)
            self._t += 1
        return self._user_event

    # noinspection SpellCheckingInspection
    def _build_color_lookup(self):
        self._background_color: pygame.Color = pygame.Color('grey10')
        self._color_lookup = {
            common.Square.NORMAL: pygame.Color('darkgrey'),
            common.Square.CLIFF: pygame.Color('red2'),
            common.Square.START: pygame.Color('yellow2'),
            common.Square.END: pygame.Color('goldenrod2'),
            common.Square.AGENT: pygame.Color('deepskyblue2')
        }

    def _load_gridworld(self):
        self._set_sizes()
        self._grid_surface.fill(self._background_color)
        for x in range(self._max_x + 1):
            for y in range(self._max_y + 1):
                square: common.Square = self._grid_world.get_square(position=common.XY(x, y))
                self._draw_square(x, y, square, self._grid_surface)
        self._copy_grid_into_background()

    def _copy_grid_into_background(self):
        self._background.blit(source=self._grid_surface, dest=(0, 0))

    def _set_sizes(self):
        # size window for track and set cell_pixels
        rows, cols = self._grid_world.max_y + 1, self._grid_world.max_x + 1
        self._cell_pixels = int(min(self._screen_height / rows, self._screen_width / cols))
        self._screen_width = cols * self._cell_pixels
        self._screen_height = rows * self._cell_pixels

        self._background = pygame.Surface(size=self.screen_size)
        self._grid_surface = pygame.Surface(size=self.screen_size)

    def _draw_square(self, x: int, y: int, square: common.Square, surface: pygame.Surface) -> pygame.Rect:
        row = self._max_y - y
        col = x

        color: pygame.Color = self._color_lookup[square]
        left: int = col * self._cell_pixels
        top: int = row * self._cell_pixels
        width: int = self._cell_pixels - 1
        height: int = self._cell_pixels - 1

        # doesn'_t like named parameters
        rect: pygame.Rect = pygame.Rect(left, top, width, height)
        pygame.draw.rect(surface, color, rect)
        return rect

    def _put_background_on_screen(self):
        self._screen.blit(source=self._background, dest=(0, 0))
        pygame.display.flip()

    def _wait_for_event_of_interest(self):
        self._user_event = common.UserEvent.NONE
        while self._user_event == common.UserEvent.NONE:
            # replaced: for event in pygame.event.get():
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                self._user_event = common.UserEvent.QUIT
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self._user_event = common.UserEvent.SPACE

    def _draw_agent(self, show_trail: bool = True):
        if not show_trail:
            self._copy_grid_into_background()
            self._put_background_on_screen()
        state: environment.State = self._episode.trajectory[self._t].state
        self._draw_agent_at_state(state)

    def _draw_agent_at_state(self, state: environment.State):
        # row, col = self._grid_world.get_index(state.x, state.y)
        # print(f"_t={self._t} x={state.x} y={state.y} row={row} col={col}")
        rect: pygame.Rect = self._draw_square(x=state.position.x, y=state.position.y,
                                              square=common.Square.AGENT, surface=self._background)
        self._screen.blit(source=self._background, dest=rect, area=rect)
        pygame.display.update(rect)
        # self.screen.blit(source=self.background, dest=(0, 0))
        # pygame.display.flip()

    def close_window(self):
        # pygame.display.quit()
        pygame.quit()
