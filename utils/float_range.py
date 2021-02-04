import math


def float_range(start: float, stop: float, step_size: float) -> list[float]:
    diff: float = stop - start
    steps: int = int(round(diff / step_size, 0))
    decimal_places: int = max(-math.floor(math.log10(0.05)), 0)
    result = [round(start + step_size*step, decimal_places) for step in range(steps+1)]
    return result
