from __future__ import annotations
from mdp.scenarios.jacks import location_summary


# ending_cars, location_summary
class LocationOutcome(dict[int, location_summary.LocationSummary]):
    pass

