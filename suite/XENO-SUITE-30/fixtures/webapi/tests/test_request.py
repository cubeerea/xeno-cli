"""Baseline behaviour of the Request value object."""

from __future__ import annotations

from webapi.request import Request


def test_method_is_upper_cased() -> None:
    request = Request(method="get", path="/health")
    assert request.method == "GET"


def test_headers_are_lower_cased_and_lookup_is_case_insensitive() -> None:
    request = Request(method="GET", path="/health", headers={"Content-Type": "text/plain"})
    assert request.headers == {"content-type": "text/plain"}
    assert request.header("CONTENT-TYPE") == "text/plain"
    assert request.header("missing") is None
    assert request.header("missing", "fallback") == "fallback"


def test_path_is_stored_verbatim() -> None:
    request = Request(method="GET", path="/search?q=cat")
    assert request.path == "/search?q=cat"


def test_defaults_are_independent_between_instances() -> None:
    first = Request(method="GET", path="/a")
    second = Request(method="GET", path="/b")
    first.context["marker"] = 1
    assert second.context == {}
    assert first.body == ""


def test_with_context_returns_a_new_request() -> None:
    request = Request(method="GET", path="/a", context={"params": {}})
    derived = request.with_context(trace="abc")
    assert derived is not request
    assert derived.context == {"params": {}, "trace": "abc"}
    assert request.context == {"params": {}}
    assert derived.path == "/a"


def test_with_body_replaces_only_the_body() -> None:
    request = Request(method="POST", path="/echo", headers={"X-A": "1"}, body="old")
    derived = request.with_body("new")
    assert derived.body == "new"
    assert derived.headers == {"x-a": "1"}
    assert request.body == "old"
