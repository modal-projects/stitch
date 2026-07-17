"""SGLangEngine harness: version stamping (pure dict mutation, no HTTP).

The /pull_weights + /update_weights_from_disk control paths hit a real engine, so they
are validated e2e; the request/response stamping is the provable-without-sglang part."""

from __future__ import annotations

from stitch.engines.sglang import SGLangEngine
from stitch.versions import VersionRef


def test_stamp_request_namespaces_by_version() -> None:
    engine = SGLangEngine("http://engine", "/ckpt")
    req: dict = {"text": "hi"}
    engine.stamp_request(req, VersionRef("r1", 7))
    assert req["extra_key"] == "wv7;r1/"  # version + run namespace, no user key
    listed: dict = {"extra_key": ["a", "b"]}
    engine.stamp_request(listed, VersionRef(None, 3))
    assert listed["extra_key"] == ["wv3;a", "wv3;b"]  # run-less, per-element


def test_stamp_response_generate_vs_openai() -> None:
    engine = SGLangEngine("http://engine", "/ckpt")
    gen: dict = {"text": "x", "meta_info": {}}
    engine.stamp_response(gen, VersionRef("r1", 4), VersionRef("r1", 5))
    assert gen["meta_info"] == {"weight_version": "4", "weight_version_start": 4, "weight_version_end": 5}
    openai: dict = {"choices": []}
    engine.stamp_response(openai, VersionRef("r1", 4), VersionRef("r1", 4))
    assert openai["weight_version_start"] == 4 and openai["weight_version_end"] == 4
    assert "meta_info" not in openai and "weight_version" not in openai


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"sglang engine harness: {len(tests)} PASS")
