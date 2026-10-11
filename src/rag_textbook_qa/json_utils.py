"""Reject ambiguous objects and non-standard numbers at JSON input boundaries."""

from __future__ import annotations

import json
import math
from typing import Any


def loads_strict(raw: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate_json_key")
            value[key] = item
        return value

    def invalid_constant(value: str) -> Any:
        raise ValueError("invalid_json_constant")

    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite_json_number")
        return number

    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=invalid_constant,
                      parse_float=finite_float)
