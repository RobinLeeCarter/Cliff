from __future__ import annotations

from typing import Optional

from mdp.model import model
from mdp.scenarios.jacks.model.environment import Environment


class Model(model.Model):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.environment: Optional[Environment] = self.environment
