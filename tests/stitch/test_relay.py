"""Editing a JSON response without decoding what the edit does not read."""

from __future__ import annotations

from typing import Any

import msgspec
import pytest

from stitch.relay import edit_json

BODY = (
    b'{"text":"ok","meta_info":{"weight_version":"3","queue_time":1.2e-5,'
    b'"routed_experts":"QUJD","output_token_sampling_mask":[[7, 3],null]}}'
)


def _stamp(data: Any) -> None:
    meta = data["meta_info"]
    meta["weight_version"] = "4"
    meta["weight_version_end"] = 5


def test_edit_changes_only_the_members_it_writes() -> None:
    assert edit_json(BODY, _stamp) == (
        b'{"text":"ok","meta_info":{"weight_version":"4","queue_time":1.2e-5,'
        b'"routed_experts":"QUJD","output_token_sampling_mask":[[7, 3],null],'
        b'"weight_version_end":5}}'
    )


def test_members_the_edit_does_not_read_stay_undecoded() -> None:
    seen: dict[str, type] = {}

    def edit(data: Any) -> None:
        meta = data["meta_info"]
        seen.update({key: type(dict.get(meta, key)) for key in meta})

    edit_json(BODY, edit)
    assert seen["routed_experts"] is msgspec.Raw
    assert seen["output_token_sampling_mask"] is msgspec.Raw


def test_arrays_decode_their_elements() -> None:
    assert edit_json(b'[1,{"a":2}]', lambda d: d[1].update(b=3)) == b'[1,{"a":2,"b":3}]'


def test_invalid_document_is_rejected() -> None:
    with pytest.raises(ValueError):
        edit_json(BODY[:-1], _stamp)
