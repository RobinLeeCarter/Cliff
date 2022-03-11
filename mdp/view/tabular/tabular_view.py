from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from abc import ABC

if TYPE_CHECKING:
    from mdp.general_controller import GeneralController

import pygame
import pygame.freetype
from mdp.view.tabular.grid_view import GridView
from mdp.view.general.general_view import GeneralView


class TabularView(GeneralView, ABC):
    def __init__(self):
        super().__init__()
        pygame_pass, pygame_fail = pygame.init()
        if pygame_fail > 0:
            raise Exception(f"{pygame_fail} pygame modules failed to load")
        self.grid_view: Optional[GridView] = None

    def set_controller(self, controller: GeneralController):
        self._controller: GeneralController = controller

    def _build(self):
        self.grid_view: GridView = self._create_grid_view()

    def _create_grid_view(self) -> GridView:
        """returns: specific GridView(grid_view_parameters)"""
        pass

    def demonstrate(self):
        if self.grid_view:
            self.grid_view.demonstrate(self._controller.new_episode_request)
        else:
            raise Exception("self.grid_view is None, possibly grid_world_ is None")
