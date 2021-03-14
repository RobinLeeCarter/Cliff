from __future__ import annotations

from mdp.scenarios.jacks.dynamics.counter import Counter
from mdp.scenarios.jacks.dynamics.location_outcome import LocationOutcome


class LocationOutcomes(Counter[LocationOutcome, float]):
    pass
