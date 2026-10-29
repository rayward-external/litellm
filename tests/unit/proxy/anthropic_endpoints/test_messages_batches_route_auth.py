"""Non-admin virtual keys must reach every /v1/messages/batches* route, and
only a full proxy admin may bypass batch ownership.

The 2026-09-24 upstream sync dropped the four batch entries from
LiteLLMRoutes.anthropic_routes, which made is_llm_api_route() False for them.
From then on every non-admin key got "Only proxy admin can be used ..." (401)
on create, list, retrieve, results, cancel and delete, while master and admin
keys kept working and hid the break.

The paths are read off the mounted APIRouter rather than listed by hand, so a
batch route added later without a matching anthropic_routes entry fails here
too.

Because these are LLM API routes, RouteChecks never runs
_check_proxy_admin_viewer_access on them. So the handler's own admin test is
the only thing standing between a read-only PROXY_ADMIN_VIEW_ONLY key and
another tenant's batch. A "proxy_admin" prefix test matched the viewer too
(codex P1 on the restore).
"""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, Request

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.anthropic_endpoints.messages_batches import (
    BEDROCK_MSGBATCH_PREFIX,
    _check_bedrock_batch_owner_tag,
    _is_proxy_admin,
    router,
)
from litellm.proxy.auth.route_checks import RouteChecks

SAMPLE_BATCH_ID = "msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d"

BATCH_ROUTES = sorted(
    {
        (method, route.path)
        for route in router.routes
        if route.path.startswith("/v1/messages/batches")
        for method in route.methods
    }
)


def _concrete(path: str) -> str:
    return path.replace("{batch_id}", SAMPLE_BATCH_ID)


def test_router_mounts_every_batch_verb():
    """Fails loudly if the router import stops yielding routes, which would
    otherwise make the parametrized tests below vacuously pass."""
    paths = {path for _, path in BATCH_ROUTES}
    assert paths == {
        "/v1/messages/batches",
        "/v1/messages/batches/{batch_id}",
        "/v1/messages/batches/{batch_id}/results",
        "/v1/messages/batches/{batch_id}/cancel",
    }
    methods = {method for method, _ in BATCH_ROUTES}
    assert {"GET", "POST", "DELETE"} <= methods


@pytest.mark.parametrize(("method", "path"), BATCH_ROUTES)
def test_is_llm_api_route(method, path):
    assert RouteChecks.is_llm_api_route(route=_concrete(path)) is True


@pytest.mark.parametrize(("method", "path"), BATCH_ROUTES)
def test_non_admin_key_passes_route_check(method, path):
    request = MagicMock(spec=Request)
    request.method = method
    request.query_params = {}

    # Must not raise (the admin-only 401) for a plain, role-less, non-admin key.
    assert (
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=None,
            _user_role=None,
            route=_concrete(path),
            request=request,
            valid_token=UserAPIKeyAuth(api_key="hashed-non-admin-key"),
            request_data={},
        )
        is None
    )


@pytest.mark.parametrize("role", list(LitellmUserRoles))
def test_only_a_full_proxy_admin_bypasses_batch_ownership(role):
    key = UserAPIKeyAuth(api_key="hashed-key", user_role=role)
    assert _is_proxy_admin(key) is (role == LitellmUserRoles.PROXY_ADMIN)


def test_admin_viewer_cannot_reach_another_owners_batch():
    foreign_id = f"{BEDROCK_MSGBATCH_PREFIX}abc123def456_0badc0de"
    viewer = UserAPIKeyAuth(
        api_key="hashed-viewer-key", user_id="viewer", user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY
    )
    with pytest.raises(HTTPException) as exc:
        _check_bedrock_batch_owner_tag(foreign_id, viewer)
    assert exc.value.status_code == 404

    admin = UserAPIKeyAuth(api_key="hashed-admin-key", user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)
    assert _check_bedrock_batch_owner_tag(foreign_id, admin) is None
