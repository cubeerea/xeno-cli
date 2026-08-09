"""ZS-22 acceptance spec: HMAC-signed bearer tokens.

A new ``webapi.auth`` subpackage signs and verifies tokens and ships a
``require_auth`` middleware.  The application must let the payload attached by
that middleware reach the handler alongside the path parameters, and the
package must re-export ``require_auth`` and ``InvalidToken``.
"""

from __future__ import annotations

from typing import cast

import pytest

import webapi
from webapi import App, InvalidToken, Request, Response, require_auth
from webapi.auth.middleware import require_auth as require_auth_direct
from webapi.auth.tokens import InvalidToken as InvalidTokenDirect
from webapi.auth.tokens import sign, verify

SECRET = "fixture-signing-secret"
PAYLOAD = {"sub": "u-1", "role": "admin"}


def _build_app(seen: list[Request]) -> App:
    app = App()
    app.use(require_auth(SECRET))

    @app.get("/me/<slot>")
    def me(request: Request) -> Response:
        seen.append(request)
        return Response.text(200, "me")

    return app


def test_sign_and_verify_round_trip() -> None:
    token = sign(PAYLOAD, SECRET)
    assert isinstance(token, str)
    assert token != ""
    assert verify(token, SECRET) == PAYLOAD


def test_signature_depends_on_the_secret() -> None:
    assert sign(PAYLOAD, SECRET) != sign(PAYLOAD, "another-secret")


def test_verify_rejects_a_tampered_token() -> None:
    token = sign(PAYLOAD, SECRET)
    with pytest.raises(InvalidToken):
        verify(token + "x", SECRET)
    with pytest.raises(InvalidToken):
        verify("x" + token, SECRET)


def test_verify_rejects_the_wrong_secret() -> None:
    token = sign(PAYLOAD, SECRET)
    with pytest.raises(InvalidToken):
        verify(token, "another-secret")


def test_verify_rejects_garbage() -> None:
    with pytest.raises(InvalidToken):
        verify("not-a-token", SECRET)


def test_invalid_token_is_the_same_class_everywhere() -> None:
    assert InvalidToken is InvalidTokenDirect
    assert issubclass(InvalidToken, Exception)
    assert "InvalidToken" in webapi.__all__


def test_require_auth_is_re_exported() -> None:
    assert require_auth is require_auth_direct
    assert "require_auth" in webapi.__all__


def test_missing_authorization_header_is_401() -> None:
    seen: list[Request] = []
    response = _build_app(seen).handle(Request("GET", "/me/2"))
    assert response.status == 401
    assert seen == []


def test_malformed_authorization_header_is_401() -> None:
    seen: list[Request] = []
    app = _build_app(seen)
    token = sign(PAYLOAD, SECRET)
    assert app.handle(Request("GET", "/me/2", headers={"Authorization": token})).status == 401
    assert seen == []


def test_bad_signature_is_401() -> None:
    seen: list[Request] = []
    app = _build_app(seen)
    token = sign(PAYLOAD, "another-secret")
    request = Request("GET", "/me/2", headers={"Authorization": f"Bearer {token}"})
    assert app.handle(request).status == 401
    assert seen == []


def test_valid_token_reaches_the_handler_with_payload_and_params() -> None:
    seen: list[Request] = []
    app = _build_app(seen)
    token = sign(PAYLOAD, SECRET)
    request = Request("GET", "/me/2", headers={"Authorization": f"Bearer {token}"})
    response = app.handle(request)
    assert response.status == 200
    assert response.body == "me"
    assert len(seen) == 1
    handler_request = seen[-1]
    assert handler_request.context["auth"] == PAYLOAD
    assert cast(dict[str, object], handler_request.context["params"]) == {"slot": "2"}
