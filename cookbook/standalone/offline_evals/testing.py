"""In-memory stand-ins for the eval router's Modal resources (single-process)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class _LocalDict:
    """Duck-types a modal.Dict: ``get``/``put``/``items`` with ``.aio`` forms."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self.get = SimpleNamespace(aio=self._get)
        self.put = SimpleNamespace(aio=self._put)
        self.items = SimpleNamespace(aio=self._items)

    async def _get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    async def _put(self, key: str, value: Any) -> None:
        self._store[key] = value

    async def _items(self) -> Any:
        # modal.Dict.items is an async generator; mirror that shape.
        for item in self._store.items():
            yield item
