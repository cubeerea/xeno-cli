"""Baseline behaviour of the Response value object."""

from __future__ import annotations

import json

from webapi.response import JSON_CONTENT_TYPE, TEXT_CONTENT_TYPE, Response


def test_text_sets_a_plain_text_content_type() -> None:
    response = Response.text(200, "pong")
    assert response.status == 200
    assert response.body == "pong"
    assert response.headers["Content-Type"] == TEXT_CONTENT_TYPE


def test_json_encodes_the_payload_deterministically() -> None:
    response = Response.json(201, {"b": 2, "a": 1})
    assert response.headers["Content-Type"] == JSON_CONTENT_TYPE
    assert response.body == '{"a":1,"b":2}'
    assert json.loads(response.body) == {"a": 1, "b": 2}


def test_header_lookup_is_case_insensitive() -> None:
    response = Response.text(200, "ok")
    assert response.header("content-type") == TEXT_CONTENT_TYPE
    assert response.header("X-Nope") is None


def test_with_header_does_not_mutate_the_original() -> None:
    original = Response.text(200, "ok")
    derived = original.with_header("X-Trace", "abc")
    assert derived.headers["X-Trace"] == "abc"
    assert "X-Trace" not in original.headers
    assert derived.body == "ok"


def test_with_header_replaces_an_existing_header_case_insensitively() -> None:
    response = Response.text(200, "ok").with_header("content-type", "text/csv")
    assert response.header("Content-Type") == "text/csv"
    assert len([key for key in response.headers if key.lower() == "content-type"]) == 1
