from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from abc import ABC

if TYPE_CHECKING:
    from mdp import common
    from mdp.controller.base_controller import BaseController

from mdp.view.base.graph2d import Graph2D
from mdp.view.base.graph3d import Graph3D


class BaseView(ABC):
    def __init__(self):
        self._controller: Optional[BaseController] = None
        self._comparison: Optional[common.Comparison] = None
        self.graph2d: Optional[Graph2D] = None
        self.graph3d: Optional[Graph3D] = None

    def set_controller(self, controller: BaseController):
        self._controller: BaseController = controller

    def build(self, comparison: common.Comparison):
        """build top-down"""
        self._comparison = comparison
        self.graph2d: Graph2D = Graph2D()
        self.graph3d: Graph3D = Graph3D()

    def demonstrate(self):
        raise Exception("demonstrate() not implemented")
