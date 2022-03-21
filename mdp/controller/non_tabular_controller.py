from __future__ import annotations
from typing import TypeVar

from mdp.controller.base_controller import BaseController
from mdp.model.non_tabular.non_tabular_model import NonTabularModel
from mdp.view.non_tabular.non_tabular_view import NonTabularView

Model = TypeVar("Model", bound=NonTabularModel)
View = TypeVar("View", bound=NonTabularView)


class NonTabularController(BaseController[Model, View]):
    pass
