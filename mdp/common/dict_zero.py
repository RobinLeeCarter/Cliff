from __future__ import annotations


class DictZero(dict):
    def __missing__(self, key) -> float:
        return 0.0
