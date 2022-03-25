from __future__ import annotations
from dataclasses import dataclass

from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction


@dataclass(frozen=True)
class PlaceholderAction(NonTabularAction):
    """
    Placeholder for when a feature is only defined in terms of State but generic definition requires an Action
    """
    def _get_categories(self) -> list:
        return []
