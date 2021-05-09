from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from abc import ABC

import pygame
import pygame.freetype

if TYPE_CHECKING:
    from mdp import common
    from mdp.controller import Controller
    from mdp.model.environment.grid_world import GridWorld

from mdp.view.graph import Graph
from mdp.view.graph3d import Graph3D
from mdp.view.grid_view import GridView


class View(ABC):
    def __init__(self):
        self._controller: Optional[Controller] = None

        pygame_pass, pygame_fail = pygame.init()
        if pygame_fail > 0:
            raise Exception(f"{pygame_fail} pygame modules failed to load")

        self.graph: Optional[Graph] = None
        self.graph3d = Graph3D()
        self.grid_view: Optional[GridView] = None

    def set_controller(self, controller_: Controller):
        self._controller: Controller = controller_

    def build(self, grid_world_: Optional[GridWorld], comparison: common.Comparison):
        self.graph: Graph = self._create_graph()
        self.graph3d: Graph3D = self._create_graph3d()
        if grid_world_:
            self.grid_view: GridView = self._create_grid_view(comparison.grid_view_parameters)
            # self.grid_view = grid_view_factory.grid_view_factory(comparison.grid_view_parameters)
            self.grid_view.set_gridworld(grid_world_)

    def _create_grid_view(self, grid_view_parameters: common.GridViewParameters) -> GridView:
        # return GridView(grid_view_parameters)
        pass

    def _create_graph(self) -> Graph:
        return Graph()

    def _create_graph3d(self) -> Graph3D:
        return Graph3D()

    def demonstrate(self):
        if self.grid_view:
            self.grid_view.demonstrate(self._controller.new_episode_request)
        else:
            raise Exception("self.grid_view is None, possibly grid_world_ is None")
