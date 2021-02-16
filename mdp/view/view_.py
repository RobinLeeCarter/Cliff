from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import controller
    from mdp.model import agent, environment
import common
from mdp.view import graph, grid_view


class View:
    def __init__(self):
        self._controller: Optional[controller.Controller] = None
        self.graph = graph.Graph()
        self.grid_view = grid_view.GridView()

    def set_controller(self, controller_: controller.Controller):
        self._controller = controller_

    def build(self, grid_world_: environment.GridWorld):
        self.grid_view.set_grid_world(grid_world_)

    def demonstrate(self):
        self.grid_view.open_window()
        running_average = 0
        count = 0
        while True:
            count += 1
            episode: agent.Episode = self._controller.get_fresh_episode()
            print(f"max_t: {episode.max_t} \t total_return: {episode.total_return:.0f}")
            running_average += (1/count) * (episode.total_return - running_average)
            print(f"count: {count} \t running_average: {running_average:.1f}")
            user_event: common.UserEvent = self.grid_view.display_episode(episode, show_trail=False)
            if user_event == common.UserEvent.QUIT:
                break
        self.grid_view.close_window()

    # def set_grid_world(self, grid_world_: environment.GridWorld):
    #     self.grid_view.set_grid_world(grid_world_)
    #
    # def grid_view_open_window(self):
    #     self.grid_view.open_window()
