"""ModalFlashPool harness: the pure URL/host normalization helpers. The Modal calls
(discover / wake / scale / gateway) are lazy and validated e2e."""

from __future__ import annotations

from stitch.pools.modal_flash import _host, _normalize_url


def test_normalize_url_adds_scheme_and_strips_slash() -> None:
    assert _normalize_url("host:8000") == "https://host:8000"
    assert _normalize_url("http://host:8000/") == "http://host:8000"
    assert _normalize_url("https://host/") == "https://host"


def test_host_reads_dict_or_attr() -> None:
    assert _host({"host": "h1"}) == "h1"
    assert _host({}) is None

    class _Container:
        host = "h2"

    assert _host(_Container()) == "h2"
    assert _host(object()) is None


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"modal_flash harness: {len(tests)} PASS")
