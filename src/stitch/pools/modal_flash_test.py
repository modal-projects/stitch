"""ModalFlashPool harness. Every method is a lazy Modal call (discover / wake / scale /
gateway), so the provable-without-Modal surface is port conformance; the live calls are
validated e2e."""

from __future__ import annotations

from stitch.pools.base import Pool
from stitch.pools.modal_flash import ModalFlashPool


def test_satisfies_pool_port() -> None:
    assert isinstance(ModalFlashPool("app", "Server"), Pool)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"modal_flash harness: {len(tests)} PASS")
