from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from abc import ABC

import pygame
import pygame.freetype

if TYPE_CHECKING:
    from mdp import common
    from mdp.general_controller import GeneralController

from mdp.view.graph import Graph
from mdp.view.graph3d import Graph3D
from mdp.view.grid_view import GridView


class GeneralView(ABC):
    def __init__(self):
        self._controller: Optional[GeneralController] = None
        self._comparison: Optional[common.Comparison] = None

        pygame_pass, pygame_fail = pygame.init()
        if pygame_fail > 0:
            raise Exception(f"{pygame_fail} pygame modules failed to load")

        self.graph: Optional[Graph] = None
        self.graph3d: Optional[Graph3D] = None
        self.grid_view: Optional[GridView] = None

    def set_controller(self, controller_: GeneralController):
        self._controller: GeneralController = controller_

    def build(self, comparison: common.Comparison):
        self._comparison = comparison
        self.graph: Graph = self._create_graph()
        self.graph3d: Graph3D = self._create_graph3d()
        self.grid_view: GridView = self._create_grid_view()

    def _create_grid_view(self) -> GridView:
        # returns: specific GridView(grid_view_parameters)
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
