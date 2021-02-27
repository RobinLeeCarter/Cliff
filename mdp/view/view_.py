from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import pygame
import pygame.freetype

if TYPE_CHECKING:
    from mdp import controller, common
    from mdp.model import environment
from mdp.view import graph, grid_view


class View:
    def __init__(self):
        self._controller: Optional[controller.Controller] = None

        pygame_pass, pygame_fail = pygame.init()
        if pygame_fail > 0:
            raise Exception(f"{pygame_fail} pygame modules failed to load")

        self.graph = graph.Graph()
        self.grid_view = grid_view.GridView()

    def set_controller(self, controller_: controller.Controller):
        self._controller = controller_

    def build(self, grid_world_: environment.GridWorld, comparison: common.Comparison):
        self.grid_view.grid_view_parameters = comparison.grid_view_parameters
        self.grid_view.set_gridworld(grid_world_)

    def demonstrate(self):
        self.grid_view.demonstrate(self._controller.new_episode_request)
