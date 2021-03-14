from __future__ import annotations
# from typing import TYPE_CHECKING


class Counter(dict):
    def __missing__(self, key) -> float:
        return 0.0
