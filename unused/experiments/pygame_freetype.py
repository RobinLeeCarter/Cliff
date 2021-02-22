from __future__ import annotations

import time
from typing import Optional, TYPE_CHECKING, Callable

import pygame
import pygame.freetype

if TYPE_CHECKING:
    from mdp.model import agent, environment
from mdp import common
from mdp.model import scenarios


class GridView:
    def __init__(self, display_v: bool = False):
        self._display_v: bool = display_v
        self._grid_world: Optional[environment.GridWorld] = None

        self._max_x: Optional[int] = None
        self._max_y: Optional[int] = None

        self._screen_width: int = 1500
        self._screen_height: int = 1000
        self._title: str = "Not implemented"
        self._cell_pixels: int = 10

        self._grid_surface: Optional[pygame.Surface] = None     # draw grid just once
        self._background: Optional[pygame.Surface] = None
        self._screen: Optional[pygame.Surface] = None

        self._background_color: Optional[pygame.Color] = None
        self._color_lookup: dict[common.Square, pygame.Color] = {}

        self._user_event: common.UserEvent = common.UserEvent.NONE

        self._t: int = 0
        self._episode: Optional[agent.Episode] = None

        self._build_color_lookup()

        t0 = time.time()
        self.font: pygame.freetype.Font = pygame.freetype.Font(None, 12)
        print('time needed for Font creation :', time.time() - t0)

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
        if self._display_v:
            self._load_gridworld()
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
                if self._display_v:
                    v = self._grid_world.v[y, x]
                    self._draw_square(x, y, square, self._grid_surface, v=v)
                else:
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

    def _draw_square(self, x: int, y: int,
                     square: common.Square,
                     surface: pygame.Surface,
                     v: Optional[float] = None
                     ) -> pygame.Rect:
        row = self._max_y - y
        col = x

        color: pygame.Color = self._color_lookup[square]
        left: int = col * self._cell_pixels
        top: int = row * self._cell_pixels
        width: int = self._cell_pixels - 1
        height: int = self._cell_pixels - 1

        # doesn'_t like named parameters
        square_rect: pygame.Rect = pygame.Rect(left, top, width, height)
        pygame.draw.rect(surface, color, square_rect)

        text: str = "12.3"

        move: common.XY = common.XY(x=-1, y=1)
        sub_rect = self._get_sub_rect(square_rect, move)
        self._center_text(surface, sub_rect, text)

        # sub_width: float = square_rect.width / 3.0
        # sub_height: float = square_rect.height / 3.0
        # sub_left: float = square_rect.left + (move.x + 1)*sub_width
        # sub_top: float = square_rect.top + (1 - move.y)*sub_height
        # sub_rect: pygame.Rect = pygame.Rect(sub_left, sub_top, sub_width, sub_height)
        # print(f"rect = {square_rect}")
        # print(f"sub  = {sub_rect}")

        # print("next...")

        # bounds = self._font.get_rect(text)
        # bounds.center = (0, 0)
        # bounds.move_ip(square_rect.center)
        # self._font.render_to(surface, bounds.topleft, text)

        # print(bounds.x, bounds.y, bounds.w, bounds.h)
        # print(bounds.center)
        # bounds.center = (0, 0)
        # print(bounds.x, bounds.y, bounds.w, bounds.h)
        # print(bounds.center)

        # bounds.center += square_rect.center
        # bounds.move_ip(square_rect.center)
        # bounds.x += square_rect.centerx
        # bounds.y += square_rect.centery
        # bounds.move(square_rect.center)
        # print(bounds.x, bounds.y, bounds.w, bounds.h)
        # print(bounds.center)
        # print(square_rect.center)

        # destination: tuple = (left, top)

        if v is not None:
            # write v in square_rect
            pass
        return square_rect

    def _get_sub_rect(self, rect: pygame.Rect, move: common.XY) -> pygame.Rect:
        sub_width: float = rect.width / 3.0
        sub_height: float = rect.height / 3.0
        sub_left: float = rect.left + (move.x + 1)*sub_width
        sub_top: float = rect.top + (1 - move.y)*sub_height
        sub_rect: pygame.Rect = pygame.Rect(sub_left, sub_top, sub_width, sub_height)
        return sub_rect

    def _center_text(self, surface: pygame.Surface, rect: pygame.Rect, text: str):
        bounds = self.font.get_rect(text)
        bounds.center = (0, 0)
        bounds.move_ip(rect.center)
        self.font.render_to(surface, bounds.topleft, text)

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


def view_test() -> bool:
    pygame_pass, pygame_fail = pygame.init()
    if pygame_fail > 0:
        raise Exception(f"{pygame_fail} pygame modules failed to load")
    # print(pygame.freetype.get_init())

    environment_parameters = common.EnvironmentParameters(
        environment_type=common.EnvironmentType.CLIFF,
        actions_list=common.ActionsList.FOUR_MOVES
    )
    cliff = scenarios.environment_factory(environment_parameters)
    my_grid_view = GridView()
    my_grid_view.set_grid_world(cliff.grid_world)
    my_grid_view.open_window()
    my_grid_view.display_and_wait()

    return True


if __name__ == '__main__':
    if view_test():
        print("Passed")
