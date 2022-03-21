from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from abc import ABC

from mdp import common

if TYPE_CHECKING:
    from mdp.controller.tabular_controller import TabularController

import pygame
import pygame.freetype
from mdp.view.tabular.grid_view import GridView
from mdp.view.base.base_view import BaseView


class TabularView(BaseView, ABC):
    def __init__(self):
        super().__init__()
        self._controller: Optional[TabularController] = None
        self.grid_view: Optional[GridView] = None

        pygame_pass, pygame_fail = pygame.init()
        if pygame_fail > 0:
            raise Exception(f"{pygame_fail} pygame modules failed to load")

    def set_controller(self, controller: TabularController):
        self._controller: TabularController = controller

    def build(self, comparison: common.Comparison):
        super().build(comparison)
        self.grid_view: GridView = self._create_grid_view()

    def _create_grid_view(self) -> GridView:
        """returns: specific GridView(grid_view_parameters)"""
        pass

    def demonstrate(self):
        if self.grid_view:
            self.grid_view.demonstrate(self._controller.new_episode_request)
        else:
            raise Exception("self.grid_view is None, possibly grid_world_ is None")
