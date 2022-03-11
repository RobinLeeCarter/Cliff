from __future__ import annotations
from typing import TypeVar

from mdp.controller.general_controller import GeneralController
from mdp.model.non_tabular.non_tabular_model import NonTabularModel
from mdp.view.non_tabular.non_tabular_view import NonTabularView

Model = TypeVar("Model", bound=NonTabularModel)
View = TypeVar("View", bound=NonTabularView)


class NonTabularController(GeneralController[Model, View]):
    pass
