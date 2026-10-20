"""Edit a JSON response without decoding the parts the edit does not read.

Responses can carry hundreds of megabytes of routed experts or top-p masks that
the sidecar never reads. Object members stay raw JSON until read, and every
member the edit leaves alone is relayed as the upstream's bytes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import msgspec

_ONE_LEVEL = (
    dict[str, msgspec.Raw]
    | list[dict[str, msgspec.Raw]]
    | str
    | int
    | float
    | bool
    | None
)


class LazyObject(dict):
    """A JSON object whose members decode when read by key."""

    def __getitem__(self, key: str) -> Any:
        value = super().__getitem__(key)
        if isinstance(value, msgspec.Raw):
            value = decode_one_level(value)
            super().__setitem__(key, value)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        return self[key] if key in self else default


def decode_one_level(raw: bytes | bytearray | msgspec.Raw) -> Any:
    """Returns a JSON value with objects as ``LazyObject`` and the elements of
    arrays decoded the same way. Raises ``ValueError`` when ``raw`` is not JSON."""
    try:
        value = msgspec.json.decode(raw, type=_ONE_LEVEL)
    except msgspec.ValidationError:
        # An array of anything but objects decodes its elements one by one.
        items = msgspec.json.decode(raw, type=list[msgspec.Raw])
        return [decode_one_level(item) for item in items]
    if isinstance(value, dict):
        return LazyObject(value)
    if isinstance(value, list):
        return [LazyObject(item) for item in value]
    return value


def edit_json(body: bytes | bytearray, edit: Callable[[Any], None]) -> bytes:
    """Returns ``body`` edited in place by ``edit``. Object members ``edit`` does
    not read keep the upstream's bytes. Raises ``ValueError`` when ``body`` is not
    JSON."""
    document = decode_one_level(body)
    edit(document)
    return msgspec.json.encode(document)
