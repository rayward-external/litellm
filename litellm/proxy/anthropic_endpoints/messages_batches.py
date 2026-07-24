"""
/v1/messages/batches — the Anthropic Message Batches API served natively by the
proxy, with per-model backend fan-out (internal fork feature).

Contract: https://platform.claude.com/docs/en/build-with-claude/batch-processing
(create / retrieve / results / cancel; list+delete are forwarded upstream).

Routing rule (stateless):
  * CREATE — if every request in the batch names the SAME model and the router
    carries a Bedrock batch deployment for it (model_name == "<model>-batch",
    litellm_params.model == "bedrock/<us. inference profile>", model_info.mode
    == "batch"), the batch becomes a Bedrock CreateModelInvocationJob (input
    JSONL staged to the deployment's s3_bucket_name). Otherwise the body is
    forwarded verbatim to the Anthropic API upstream.
  * RETRIEVE / RESULTS / CANCEL — dispatched purely on the id prefix:
    "msgbatch_bedrock_<jobid>" ids are Bedrock jobs (job ARN reconstructed from
    the deployment's aws_batch_role_arn account + region — no DB row needed);
    anything else is forwarded upstream untouched.

The Bedrock job is mapped onto the Anthropic MessageBatch shape:
  Submitted/Validating/Scheduled/InProgress -> in_progress
  Stopping                                  -> canceling
  Completed/PartiallyCompleted/Failed/Stopped/Expired -> ended
Records missing from the output JSONL are emitted as result type "expired"
(PartiallyCompleted), "canceled" (Stopped) or "errored" (Failed) — the input
JSONL still in S3 supplies the full custom_id set, keeping this stateless.

Authorization model (beyond user_api_key_auth on every route):
  * CREATE checks every distinct requests[].params.model against the key's
    model permissions (can_key_call_model; the "<model>-batch" alias also
    satisfies the check for the Bedrock leg).
  * Bedrock batch ids embed an OWNER TAG (msgbatch_bedrock_<jobid>_<owner8>,
    owner8 = sha256 of team_id|user_id|api_key)[:8]) — retrieve/results/
    cancel/delete 404 unless the caller's tag matches or the caller is a
    proxy admin. Stateless, so ids stay self-contained.
  * Upstream batches share the proxy's single Anthropic workspace — the same
    visibility semantics Anthropic gives keys within one workspace, and the
    same posture as the existing /anthropic passthrough. LIST must therefore
    never forward that shared workspace to a non-admin key (it is the
    enumeration vector). Admin keys get the verbatim upstream forward;
    non-admin keys get an OWNER-SCOPED listing derived from the billing ledger
    (see the list route) instead, so the enumeration stays closed.

Spend tracking: every batch this route creates (both legs) is registered in
LiteLLM_ManagedObjectTable with a unified batch id, so the stock CheckBatchCost
poller retrieves the job at completion, prices it from the deployment's
model_info batch-rate fields (input/output_cost_per_token_batches), and writes
spend attributed to the submitting key/user/team (attribution rides
file_object.litellm_attribution; see CheckBatchCost._get_job_attribution).
With ANTHROPIC_BATCHES_REQUIRE_BILLING=true the create is refused (and the
just-created job stopped/canceled) if the billing row cannot be stored — an
unbillable batch never runs.

LIST (GET /v1/messages/batches): admin keys forward upstream (the full shared
workspace); non-admin keys get an owner-scoped listing rebuilt from the billing
ledger (LiteLLM_ManagedObjectTable rows this route registered), which — unlike
the old upstream-only admin view — also surfaces the caller's Bedrock jobs. It
is a ledger view, so a batch's processing_status reflects what the CheckBatchCost
poller has recorded, not a live provider poll; RETRIEVE /{id} gives the live
status.
"""

import datetime
import hashlib
import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote as _url_quote

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from litellm._logging import verbose_proxy_logger
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.types.llms.custom_http import httpxSpecialProvider

router = APIRouter()

from litellm.litellm_core_utils.cloud_storage_security import (
    BEDROCK_MANAGED_S3_OUTPUT_PREFIX,
    BEDROCK_MANAGED_S3_UPLOAD_PREFIX,
)

BEDROCK_MSGBATCH_PREFIX = "msgbatch_bedrock_"
_ANTHROPIC_VERSION_DEFAULT = "2023-06-01"
_CUSTOM_ID_PATTERN = re.compile(r"\A[a-zA-Z0-9_-]{1,64}\Z")
# Staged under LiteLLM's managed S3 namespaces so the CheckBatchCost poller's
# output download passes validate_managed_cloud_file_id (reads outside the
# managed prefixes are rejected: "file_id must reference a LiteLLM-managed
# storage object" — hit live 2026-07-18). The route's own results/delete
# handlers reconstruct keys from the job's stored data config, so pre-existing
# jobs under other prefixes still serve results — they just can't be priced.
_S3_INPUT_PREFIX = f"{BEDROCK_MANAGED_S3_UPLOAD_PREFIX}anthropic-messages-batches/"
_S3_OUTPUT_PREFIX = f"{BEDROCK_MANAGED_S3_OUTPUT_PREFIX}anthropic-messages-batches/"
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _anthropic_error(status_code: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"type": "error", "error": {"type": error_type, "message": message}},
    )


def _rfc3339(dt: datetime.datetime) -> str:
    return dt.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


# ── Bedrock deployment discovery ─────────────────────────────────────────────


def _get_llm_router():
    from litellm.proxy.proxy_server import llm_router

    return llm_router


def _find_bedrock_batch_deployment(model: str) -> Optional[Dict[str, Any]]:
    """Return the router deployment dict backing Bedrock batch for `model`.

    Accepts either the bare client-facing name ("claude-opus-4-6") or the
    explicit "-batch" alias, so users of the OpenAI-shape flow can reuse the
    same name here.
    """
    llm_router = _get_llm_router()
    if llm_router is None:
        return None
    wanted = {f"{model}-batch", model}
    for deployment in llm_router.get_model_list() or []:
        if deployment.get("model_name") not in wanted:
            continue
        litellm_params = deployment.get("litellm_params") or {}
        model_info = deployment.get("model_info") or {}
        if str(litellm_params.get("model", "")).startswith("bedrock/") and model_info.get("mode") == "batch":
            return deployment
    return None


def _any_bedrock_batch_deployment() -> Optional[Dict[str, Any]]:
    """First Bedrock batch deployment — supplies region/account/bucket for
    id-only operations (retrieve/results/cancel). Assumes one AWS account +
    region for all Bedrock batch deployments (true for this stack; documented)."""
    llm_router = _get_llm_router()
    if llm_router is None:
        return None
    for deployment in llm_router.get_model_list() or []:
        litellm_params = deployment.get("litellm_params") or {}
        model_info = deployment.get("model_info") or {}
        if str(litellm_params.get("model", "")).startswith("bedrock/") and model_info.get("mode") == "batch":
            return deployment
    return None


def _bedrock_context(deployment: Dict[str, Any]) -> Dict[str, str]:
    params = deployment.get("litellm_params") or {}
    role_arn = params.get("aws_batch_role_arn") or os.getenv("AWS_BATCH_ROLE_ARN") or ""
    account_match = re.search(r"\Aarn:aws:iam::(\d+):role/", role_arn)
    bucket = params.get("s3_bucket_name") or os.getenv("AWS_S3_BUCKET_NAME") or ""
    region = params.get("s3_region_name") or params.get("aws_region_name") or os.getenv("AWS_REGION_NAME") or ""
    if not (account_match and bucket and region and role_arn):
        raise HTTPException(
            status_code=500,
            detail={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "Bedrock batch deployment is missing aws_batch_role_arn / s3_bucket_name / region configuration.",
                },
            },
        )
    return {
        "account_id": account_match.group(1),
        "bucket": bucket,
        "region": region,
        "batch_role_arn": role_arn,
        "model_id": str(params.get("model", "")).removeprefix("bedrock/"),
        "params": params,  # type: ignore[dict-item]
    }


# ── SigV4 helpers (same pattern as bedrock files/batches transformations) ────

_aws = BaseAWSLLM()


def _sign(
    method: str,
    url: str,
    body: Optional[str],
    service: str,
    aws_params: Dict[str, Any],
    region: str,
) -> Tuple[Dict[str, str], Optional[bytes]]:
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    credentials = _aws.get_credentials(
        aws_access_key_id=aws_params.get("aws_access_key_id"),
        aws_secret_access_key=aws_params.get("aws_secret_access_key"),
        aws_session_token=aws_params.get("aws_session_token"),
        aws_region_name=region,
        aws_session_name=aws_params.get("aws_session_name"),
        aws_profile_name=aws_params.get("aws_profile_name"),
        aws_role_name=aws_params.get("aws_role_name"),
        aws_web_identity_token=aws_params.get("aws_web_identity_token"),
        aws_sts_endpoint=aws_params.get("aws_sts_endpoint"),
    )
    payload = body.encode("utf-8") if body is not None else b""
    headers = {"x-amz-content-sha256": hashlib.sha256(payload).hexdigest()}
    if service == "bedrock" and body is not None:
        headers["Content-Type"] = "application/json"
    aws_request = AWSRequest(method=method, url=url, data=payload or None, headers=headers)
    SigV4Auth(credentials, service, region).add_auth(aws_request)
    return dict(aws_request.headers), (payload or None)


async def _aws_call(
    method: str,
    url: str,
    body: Optional[str],
    service: str,
    aws_params: Dict[str, Any],
    region: str,
) -> httpx.Response:
    headers, payload = _sign(method, url, body, service, aws_params, region)
    client = get_async_httpx_client(llm_provider=httpxSpecialProvider.PassThroughEndpoint, params={"timeout": 120.0})
    return await client.client.request(method, url, headers=headers, content=payload)


# ── Upstream (api.anthropic.com) forwarding ──────────────────────────────────


def _upstream_base() -> str:
    return os.getenv("ANTHROPIC_API_BASE") or os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"


def _owner_tag(user_api_key_dict: UserAPIKeyAuth) -> str:
    """Stable 8-hex owner fingerprint embedded in Bedrock batch ids.

    Prefers team_id (survives key rotation within a team), then user_id, then
    the key hash itself as a last resort.
    """
    basis = (
        user_api_key_dict.team_id
        or user_api_key_dict.user_id
        or user_api_key_dict.api_key  # already a hash in UserAPIKeyAuth
        or ""
    )
    return hashlib.sha256(str(basis).encode()).hexdigest()[:8]


def _is_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> bool:
    role = getattr(user_api_key_dict, "user_role", None)
    return str(getattr(role, "value", role) or "").startswith("proxy_admin")


def _split_bedrock_batch_id(batch_id: str) -> Tuple[str, Optional[str]]:
    """msgbatch_bedrock_<jobid>[_<owner8>] -> (jobid, owner8|None).

    Ids minted before owner tags existed have no suffix (jobid itself is
    alphanumeric, so the last "_" separates the tag unambiguously).
    """
    token = batch_id.removeprefix(BEDROCK_MSGBATCH_PREFIX)
    if "_" in token:
        job_id, owner8 = token.rsplit("_", 1)
        return job_id, owner8
    return token, None


def _batch_not_found(batch_id: str) -> HTTPException:
    # 404 (not 403) so foreign ids don't leak existence.
    return HTTPException(
        status_code=404,
        detail={"type": "error", "error": {"type": "not_found_error", "message": f"batch {batch_id} not found"}},
    )


def _check_bedrock_batch_owner_tag(batch_id: str, user_api_key_dict: UserAPIKeyAuth) -> None:
    """Legacy id-embedded ownership tag check — only used when no billing row
    exists to bind the job to an owner server-side. A missing owner suffix is
    NOT a free pass: stripping "_<owner8>" off a leaked id must not bypass
    ownership. Untagged ids (minted before owner tags existed) are admin-only."""
    _job_id, owner8 = _split_bedrock_batch_id(batch_id)
    if _is_proxy_admin(user_api_key_dict):
        return
    if owner8 is None or owner8 != _owner_tag(user_api_key_dict):
        raise _batch_not_found(batch_id)


async def _check_bedrock_batch_owner(batch_id: str, user_api_key_dict: UserAPIKeyAuth) -> None:
    """Server-side ownership for Bedrock batch ids.

    The id-embedded tag alone is forgeable: an attacker can splice a leaked
    job id together with the tag from THEIR OWN batch and pass a pure
    tag-vs-caller comparison (codex P1 round 2). The billing row binds job id
    to owner server-side, so when one exists it is authoritative. Row-less
    ids (pre-billing batches / no-DB deployments) fall back to the tag check
    with the billing-required posture deciding strictness."""
    if _is_proxy_admin(user_api_key_dict):
        return
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        _check_bedrock_batch_owner_tag(batch_id, user_api_key_dict)
        return
    row = await _get_billing_row(batch_id)
    if row is not None:
        if not _row_matches_caller(row, user_api_key_dict):
            raise _batch_not_found(batch_id)
        return
    if _billing_required():
        # Every batch created under the fail-closed gate has a row; a
        # row-less id is foreign, pre-billing, or a DB read failure — 404.
        raise _batch_not_found(batch_id)
    _check_bedrock_batch_owner_tag(batch_id, user_api_key_dict)


async def _check_model_access(models: List[str], user_api_key_dict: UserAPIKeyAuth) -> Optional[JSONResponse]:
    """Enforce the key's model permissions on batch creation (the request
    carries models only in requests[].params.model, which the generic proxy
    auth never inspects). The "<model>-batch" alias also satisfies the check
    so keys provisioned against the OpenAI-shape batch name work here too."""
    from litellm.proxy.auth.auth_checks import can_key_call_model
    from litellm.proxy.proxy_server import llm_model_list

    llm_router = _get_llm_router()
    for model in sorted(set(models)):
        allowed = False
        for candidate in (model, f"{model}-batch"):
            try:
                await can_key_call_model(
                    model=candidate,
                    llm_model_list=llm_model_list,
                    valid_token=user_api_key_dict,
                    llm_router=llm_router,
                )
                allowed = True
                break
            except Exception:
                continue
        if not allowed:
            return _anthropic_error(403, "permission_error", f"key is not allowed to call model {model!r}")
    return None


def _upstream_headers(request: Request) -> Dict[str, str]:
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        passthrough_endpoint_router,
    )

    api_key = passthrough_endpoint_router.get_credentials(
        custom_llm_provider="anthropic", region_name=None
    ) or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail={
                "type": "error",
                "error": {"type": "api_error", "message": "No Anthropic credential configured on the proxy."},
            },
        )
    headers = {
        "x-api-key": api_key,
        "anthropic-version": request.headers.get("anthropic-version", _ANTHROPIC_VERSION_DEFAULT),
        "content-type": "application/json",
    }
    # Contract-relevant client headers survive the forward.
    for passthrough_header in ("anthropic-beta", "anthropic-user-profile-id"):
        value = request.headers.get(passthrough_header)
        if value:
            headers[passthrough_header] = value
    return headers


async def _forward_upstream(
    request: Request,
    method: str,
    path: str,
    body: Optional[bytes] = None,
) -> Response:
    url = f"{_upstream_base()}{path}"
    client = get_async_httpx_client(llm_provider=httpxSpecialProvider.PassThroughEndpoint, params={"timeout": 600.0})
    upstream = await client.client.request(method, url, headers=_upstream_headers(request), content=body)
    media_type = upstream.headers.get("content-type", "application/json")
    content = upstream.content
    # Rewrite results_url (and any data[].results_url on list responses) to
    # point back at THIS gateway: the Anthropic SDKs follow results_url as an
    # ABSOLUTE URL (no base_url re-substitution — verified live), so a verbatim
    # upstream value would send the client's GATEWAY key to api.anthropic.com
    # (401 invalid x-api-key). Our /results route forwards with the proxy's
    # upstream credential instead.
    if upstream.status_code == 200 and "json" in media_type:
        try:
            payload = json.loads(content)
            base = _results_base_url(request)

            def _rewrite(obj: Dict[str, Any]) -> None:
                results_url = obj.get("results_url")
                batch_id = obj.get("id")
                if results_url and batch_id:
                    obj["results_url"] = f"{base}/v1/messages/batches/{batch_id}/results"

            if isinstance(payload, dict):
                _rewrite(payload)
                for item in payload.get("data") or []:
                    if isinstance(item, dict):
                        _rewrite(item)
                content = json.dumps(payload).encode()
        except (ValueError, TypeError):
            pass
    response_headers = {name: value for name in ("request-id", "retry-after") if (value := upstream.headers.get(name))}
    response_headers.update(
        {name: value for name, value in upstream.headers.items() if name.lower().startswith("anthropic-ratelimit-")}
    )
    return Response(content=content, status_code=upstream.status_code, media_type=media_type, headers=response_headers)


# ── Billing: managed-object registration (CheckBatchCost pickup) ─────────────

_REQUIRE_BILLING_ENV = "ANTHROPIC_BATCHES_REQUIRE_BILLING"


def _billing_required() -> bool:
    return os.getenv(_REQUIRE_BILLING_ENV, "").strip().lower() in ("1", "true", "yes")


def _unified_batch_object_id(router_model_id: str, provider_batch_id: str) -> str:
    """The base64 unified id CheckBatchCost decodes back into
    (deployment model_info.id, raw provider batch id)."""
    import base64

    from litellm.types.utils import SpecialEnums

    raw = SpecialEnums.LITELLM_MANAGED_BATCH_COMPLETE_STR.value.format(router_model_id, provider_batch_id)
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def _find_anthropic_deployment_model_id(model: str) -> Optional[str]:
    """Deployment id (model_info.id) CheckBatchCost routes its retrieve through.

    Preference order:
      1. The Anthropic-provider deployment serving `model` itself.
      2. ANY Anthropic-provider deployment — retrieve only needs workspace
         credentials (the proxy runs one shared Anthropic workspace), and the
         per-record pricing uses each result row's own model field, so a
         borrowed deployment still prices correctly.
    Returns None when the router carries no Anthropic deployment at all: the
    "anthropic/<model>" provider-string convention is NOT usable here —
    llm_router.aretrieve_batch cannot resolve it (verified live 2026-07-18:
    the poller skips such rows forever), so registering it would satisfy
    ANTHROPIC_BATCHES_REQUIRE_BILLING with a row that never bills (codex P1).
    """
    llm_router = _get_llm_router()
    fallback_any: Optional[str] = None
    for deployment in (llm_router.get_model_list() or []) if llm_router is not None else []:
        litellm_params = deployment.get("litellm_params") or {}
        # Both provider spellings occur in the wild: model: "anthropic/<m>",
        # or model: "<m>" + custom_llm_provider: "anthropic" (this stack).
        is_anthropic = str(litellm_params.get("model", "")).startswith("anthropic/") or (
            litellm_params.get("custom_llm_provider") == "anthropic"
        )
        if not is_anthropic:
            continue
        model_id = (deployment.get("model_info") or {}).get("id")
        if not model_id:
            continue
        if deployment.get("model_name") == model:
            return str(model_id)
        if fallback_any is None:
            fallback_any = str(model_id)
    return fallback_any


def _billing_preflight(router_model_id: Optional[str]) -> Optional[JSONResponse]:
    """Fail-closed check BEFORE any provider-side job is created: with
    ANTHROPIC_BATCHES_REQUIRE_BILLING=true, refusing up-front closes most of
    the created-but-unbillable window that post-create cancellation can only
    shrink, never eliminate (codex P1)."""
    if not _billing_required():
        return None
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None or not router_model_id:
        verbose_proxy_logger.error(
            "messages/batches billing: preflight refused batch create (prisma=%s router_model_id=%r)",
            prisma_client is not None,
            router_model_id,
        )
        return _anthropic_error(503, "api_error", "batch accounting is unavailable; the batch was not accepted")
    return None


async def _record_batch_for_billing(
    *,
    provider_batch_id: str,
    router_model_id: Optional[str],
    client_batch_id: str,
    model_name: str,
    total_records: int,
    user_api_key_dict: UserAPIKeyAuth,
    mixed_models: bool = False,
) -> bool:
    """Register the batch in LiteLLM_ManagedObjectTable so CheckBatchCost
    prices it at completion and writes key/user/team-attributed spend.

    Returns False when the row could not be stored (no DB, no resolvable
    deployment id, or the write failed) — the caller decides whether that is
    fatal via ANTHROPIC_BATCHES_REQUIRE_BILLING."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        verbose_proxy_logger.error(
            "messages/batches billing: no database configured; batch %s will not be cost-tracked",
            client_batch_id,
        )
        return False
    if not router_model_id:
        verbose_proxy_logger.error(
            "messages/batches billing: no router deployment id for model %r; batch %s will not be cost-tracked",
            model_name,
            client_batch_id,
        )
        return False

    unified_object_id = _unified_batch_object_id(router_model_id, provider_batch_id)
    file_object = {
        "id": client_batch_id,
        # Duplicate of `id`, but under a key CheckBatchCost._finalized_file_object
        # PRESERVES: at finalization the poller rewrites file_object from the
        # provider response, overwriting `id` with the provider batch id / ARN
        # (the owner-tagged Bedrock client id would otherwise be lost). The
        # owner-scoped LIST reads this to return an id that still works with
        # GET /v1/messages/batches/{id}, and to positively tag the row as an
        # Anthropic message batch (vs an OpenAI-dialect /v1/batches Bedrock job).
        "litellm_client_batch_id": client_batch_id,
        "object": "batch",
        "status": "validating",
        "model": model_name,
        # Mixed-model batches must not be priced with the (arbitrary first)
        # deployment's custom batch rates — the poller strips model_info for
        # these rows so each result row prices by its own model (codex P1).
        "mixed_models": mixed_models,
        "total_records": total_records,
        # Read by CheckBatchCost._get_job_attribution — the table itself only
        # carries created_by/team_id, not the submitting key.
        "litellm_attribution": {
            "user_api_key": user_api_key_dict.api_key,  # already the hash in UserAPIKeyAuth
            "user_api_key_user_id": user_api_key_dict.user_id,
            "user_api_key_team_id": user_api_key_dict.team_id,
            "user_api_key_end_user_id": user_api_key_dict.end_user_id,
            "user_api_key_alias": user_api_key_dict.key_alias,
        },
        "created_at": _rfc3339(datetime.datetime.now(datetime.timezone.utc)),
    }
    try:
        await prisma_client.db.litellm_managedobjecttable.upsert(
            where={"unified_object_id": unified_object_id},
            data={
                "create": {
                    "unified_object_id": unified_object_id,
                    "file_object": json.dumps(file_object),
                    "model_object_id": provider_batch_id,
                    "file_purpose": "batch",
                    "created_by": user_api_key_dict.user_id,
                    "team_id": user_api_key_dict.team_id,
                    # Indexed owner column: the owner-scoped LIST matches on it in
                    # SQL (single keyset query) instead of scanning + filtering the
                    # attribution JSON in Python. Its presence (NOT NULL) is also
                    # the message-batch route discriminator — OpenAI-dialect
                    # /v1/batches rows never set it. Same value the attribution
                    # stash carries: the submitting virtual key's hash.
                    "owner_key": user_api_key_dict.api_key,
                    "updated_by": user_api_key_dict.user_id,
                    "status": "validating",
                },
                "update": {
                    "file_object": json.dumps(file_object),
                    "status": "validating",
                    "owner_key": user_api_key_dict.api_key,
                    "updated_by": user_api_key_dict.user_id,
                },
            },
        )
        return True
    except Exception:
        verbose_proxy_logger.exception(
            "messages/batches billing: failed to record batch %s for cost tracking", client_batch_id
        )
        return False


async def _get_billing_row(batch_id: str, *, raise_on_error: bool = False) -> Optional[Any]:
    """The ManagedObjectTable row registered for `batch_id` (Bedrock rows key
    on the jobArn — matched via endswith on the job id — upstream rows on the
    raw msgbatch id). None when the DB is absent or no row exists; DB errors
    return None unless raise_on_error (callers whose decision must fail
    CLOSED on lookup errors — e.g. the delete gate — set it)."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return None
    try:
        if batch_id.startswith(BEDROCK_MSGBATCH_PREFIX):
            job_id, _owner8 = _split_bedrock_batch_id(batch_id)
            return await prisma_client.db.litellm_managedobjecttable.find_first(
                where={"model_object_id": {"endswith": f"/model-invocation-job/{job_id}"}}
            )
        return await prisma_client.db.litellm_managedobjecttable.find_first(where={"model_object_id": batch_id})
    except Exception:
        verbose_proxy_logger.exception("messages/batches: billing-row lookup failed for %s", batch_id)
        if raise_on_error:
            raise
        return None


def _row_attribution(row: Any) -> Dict[str, Any]:
    file_object = getattr(row, "file_object", None)
    for _ in range(2):
        if not isinstance(file_object, str):
            break
        try:
            file_object = json.loads(file_object)
        except ValueError:
            return {}
    if isinstance(file_object, dict):
        attribution = file_object.get("litellm_attribution")
        if isinstance(attribution, dict):
            return attribution
    return {}


def _row_matches_caller(row: Any, user_api_key_dict: UserAPIKeyAuth) -> bool:
    attribution = _row_attribution(row)
    return bool(
        (user_api_key_dict.api_key and attribution.get("user_api_key") == user_api_key_dict.api_key)
        or (user_api_key_dict.team_id and attribution.get("user_api_key_team_id") == user_api_key_dict.team_id)
        or (user_api_key_dict.user_id and attribution.get("user_api_key_user_id") == user_api_key_dict.user_id)
        or (user_api_key_dict.team_id and getattr(row, "team_id", None) == user_api_key_dict.team_id)
        or (user_api_key_dict.user_id and getattr(row, "created_by", None) == user_api_key_dict.user_id)
    )


async def _check_upstream_batch_owner(batch_id: str, user_api_key_dict: UserAPIKeyAuth) -> None:
    """Per-tenant isolation for upstream (api.anthropic.com) batches: the
    proxy holds ONE workspace credential, so without this check any caller
    who learns another tenant's msgbatch id could read or mutate it through
    the forward (codex P1). Ownership comes from the billing row's
    attribution; rows are written for every batch this route creates.

    Admin keys bypass. Deployments without a DB keep the historical
    shared-workspace posture. Row-less ids: with billing REQUIRED every
    batch has a row, so no-row means foreign/pre-billing — 404; with billing
    optional a registration failure is survivable by design, so the batch
    must stay usable by everyone sharing the workspace (codex P1 round 2:
    the owner would otherwise be locked out of their own batch)."""
    from litellm.proxy.proxy_server import prisma_client

    if _is_proxy_admin(user_api_key_dict) or prisma_client is None:
        return
    row = await _get_billing_row(batch_id)
    if row is None:
        if _billing_required():
            raise _batch_not_found(batch_id)
        return
    if not _row_matches_caller(row, user_api_key_dict):
        raise _batch_not_found(batch_id)


_FINALIZED_ROW_STATUSES = {"complete", "completed", "failed", "expired", "cancelled", "stale_expired"}


async def _unbilled_delete_block(batch_id: str, user_api_key_dict: UserAPIKeyAuth) -> Optional[JSONResponse]:
    """Refuse to delete a batch whose cost has not been finalized: deleting
    removes the provider-side output the poller prices from, so an early
    delete would erase the evidence and the spend (codex P1).

    batch_processed=True alone is NOT finalization — the poller sets it while
    merely holding the pricing claim, so the status must also be terminal
    (codex P1 round 2). Lookup errors fail CLOSED when billing is required
    (fail-open would let a DB outage authorize deleting unpriced output);
    admins bypass entirely."""
    if _is_proxy_admin(user_api_key_dict):
        return None
    try:
        row = await _get_billing_row(batch_id, raise_on_error=_billing_required())
    except Exception:
        return _anthropic_error(500, "api_error", "batch accounting is unavailable; retry the delete shortly")
    if row is None:
        return None
    finalized = (
        getattr(row, "batch_processed", False) is True
        and str(getattr(row, "status", "") or "") in _FINALIZED_ROW_STATUSES
    )
    if not finalized:
        return _anthropic_error(
            400,
            "invalid_request_error",
            "batch cost has not been finalized yet; retry the delete after the next billing cycle",
        )
    return None


# ── Bedrock <-> MessageBatch mapping ─────────────────────────────────────────


def _job_arn(job_id: str, ctx: Dict[str, str]) -> str:
    return f"arn:aws:bedrock:{ctx['region']}:{ctx['account_id']}:model-invocation-job/{job_id}"


def _job_url(job_id: str, ctx: Dict[str, str], suffix: str = "") -> str:
    quoted = httpx.QueryParams()  # noqa: F841 — keep httpx import obvious
    from urllib.parse import quote

    return (
        f"https://bedrock.{ctx['region']}.amazonaws.com/model-invocation-job/"
        f"{quote(_job_arn(job_id, ctx), safe='')}{suffix}"
    )


_ENDED_STATUSES = {"Completed", "PartiallyCompleted", "Failed", "Stopped", "Expired"}


def _map_job_to_message_batch(job: Dict[str, Any], batch_id: str, results_base_url: str) -> Dict[str, Any]:
    status = job.get("status", "Submitted")
    total = int(job.get("totalRecordCount") or 0)
    processed = int(job.get("processedRecordCount") or 0)
    succeeded = int(job.get("successRecordCount") or 0)
    errored = int(job.get("errorRecordCount") or 0)
    remainder = max(total - processed, 0)

    counts = {"processing": 0, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}
    if status == "Stopping":
        # Anthropic contract: terminal counters stay 0 until the WHOLE batch
        # ends — everything still reads as processing while canceling.
        processing_status = "canceling"
        counts.update(processing=total)
    elif status in _ENDED_STATUSES:
        processing_status = "ended"
        counts.update(succeeded=succeeded, errored=errored)
        if status == "PartiallyCompleted":
            counts["expired"] = remainder
        elif status == "Stopped":
            counts["canceled"] = remainder
        elif status == "Expired":
            # Normally the job never started (counters all 0 -> whole batch
            # expired); if any records did land, keep sum(counts) == total.
            counts["expired"] = max(total - succeeded - errored, 0)
        elif status == "Failed":
            counts["errored"] = max(total - succeeded, 0)
        elif status == "Completed" and remainder:
            # Shouldn't happen (Completed implies processed == total), but AWS
            # documents up-to-1-minute counter lag — keep the Anthropic
            # invariant sum(counts) == total; the results endpoint emits the
            # same records as errored ("result missing from batch output").
            counts["errored"] = errored + remainder
    else:  # Submitted / Validating / Scheduled / InProgress
        # Same contract note as Stopping: all requests remain "processing"
        # until the batch ends (Bedrock's live counters are NOT exposed —
        # Anthropic keeps terminal counters at 0 pre-end).
        processing_status = "in_progress"
        counts.update(processing=total)

    submit_time = job.get("submitTime")
    end_time = job.get("endTime")
    created_at = submit_time or _rfc3339(datetime.datetime.now(datetime.timezone.utc))
    expires_at = job.get("jobExpirationTime") or created_at
    ended = processing_status == "ended"
    return {
        "id": batch_id,
        "type": "message_batch",
        "processing_status": processing_status,
        "request_counts": counts,
        "created_at": created_at,
        "expires_at": expires_at,
        "ended_at": end_time if ended else None,
        "archived_at": None,
        "cancel_initiated_at": _rfc3339(datetime.datetime.now(datetime.timezone.utc))
        if status in ("Stopping",)
        else None,
        "results_url": f"{results_base_url}/v1/messages/batches/{batch_id}/results" if ended else None,
    }


def _bedrock_error_to_anthropic(error: Dict[str, Any]) -> Dict[str, Any]:
    code = error.get("errorCode")
    message = str(error.get("errorMessage") or "batch record failed")
    if code in (400, "400"):
        error_type = "invalid_request_error"
    elif code in (429, "429"):
        error_type = "rate_limit_error"
    else:
        error_type = "api_error"
    return {
        "type": "errored",
        "error": {"type": "error", "request_id": None, "error": {"type": error_type, "message": message}},
    }


# ── Routes ───────────────────────────────────────────────────────────────────


@router.post(
    "/v1/messages/batches",
    tags=["[beta] Anthropic `/v1/messages/batches`"],
    dependencies=[Depends(user_api_key_auth)],
)
async def create_message_batch(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    body = await _read_request_body(request=request)
    requests_list = body.get("requests")
    if not isinstance(requests_list, list) or not requests_list:
        return _anthropic_error(400, "invalid_request_error", "requests: must be a non-empty array")

    models = {str((r.get("params") or {}).get("model", "")) for r in requests_list}
    denied = await _check_model_access(sorted(models), user_api_key_dict)
    if denied is not None:
        return denied
    single_model = next(iter(models)) if len(models) == 1 else None
    # Bedrock enforces a fixed 100-record minimum per invocation job, so small
    # batches take the Anthropic-native leg even for Bedrock-batch-backed
    # models (same models, same 50% batch rate — just a different backend).
    deployment = _find_bedrock_batch_deployment(single_model) if single_model and len(requests_list) >= 100 else None
    if deployment is None:
        # Mixed-model batches, sub-100-record batches, and models without a
        # Bedrock batch backend run on the Anthropic API upstream (which
        # supports every claude model).
        billing_model = sorted(models)[0]
        billing_model_id = _find_anthropic_deployment_model_id(billing_model)
        refused = _billing_preflight(billing_model_id)
        if refused is not None:
            # Refuse BEFORE the upstream batch exists — no cancellation
            # cleanup, no unbillable-job window.
            return refused
        response = await _forward_upstream(request, "POST", "/v1/messages/batches", json.dumps(body).encode())
        upstream_batch_id: Optional[str] = None
        upstream_total = len(requests_list)
        if response.status_code == 200:
            try:
                payload = json.loads(response.body)
                if isinstance(payload.get("id"), str) and payload["id"].startswith("msgbatch_"):
                    upstream_batch_id = payload["id"]
            except (ValueError, TypeError):
                pass
        if upstream_batch_id is not None:
            # The deployment only supplies retrieve credentials (one shared
            # workspace); mixed batches are flagged so pricing never uses its
            # custom batch rates.
            recorded = await _record_batch_for_billing(
                provider_batch_id=upstream_batch_id,
                router_model_id=billing_model_id,
                client_batch_id=upstream_batch_id,
                model_name=billing_model,
                total_records=upstream_total,
                user_api_key_dict=user_api_key_dict,
                mixed_models=len(models) > 1,
            )
            if not recorded and _billing_required():
                # Refuse + cancel the just-created upstream batch, and VERIFY
                # the cancel took — an ignored failure leaves a running,
                # untracked batch (codex P1).
                cancel_confirmed = False
                try:
                    client = get_async_httpx_client(
                        llm_provider=httpxSpecialProvider.PassThroughEndpoint, params={"timeout": 60.0}
                    )
                    cancel_response = await client.post(
                        f"{_upstream_base()}/v1/messages/batches/{_url_quote(upstream_batch_id, safe='')}/cancel",
                        headers=_upstream_headers(request),
                    )
                    if cancel_response.status_code == 200:
                        cancel_status = cancel_response.json().get("processing_status")
                        cancel_confirmed = cancel_status in ("canceling", "ended")
                except Exception:
                    verbose_proxy_logger.exception(
                        "messages/batches billing: cancel of unbillable upstream batch %s failed", upstream_batch_id
                    )
                if not cancel_confirmed:
                    verbose_proxy_logger.critical(
                        "messages/batches billing: upstream batch %s is RUNNING WITHOUT COST TRACKING "
                        "(billing row not stored, cancel unconfirmed) — reconcile manually",
                        upstream_batch_id,
                    )
                return _anthropic_error(503, "api_error", "batch accounting is unavailable; the batch was not accepted")
        return response

    # ── Bedrock leg ──
    custom_ids: List[str] = []
    records: List[str] = []
    for item in requests_list:
        custom_id = str(item.get("custom_id", ""))
        params = dict(item.get("params") or {})
        if not _CUSTOM_ID_PATTERN.match(custom_id):
            return _anthropic_error(
                400, "invalid_request_error", f"custom_id must match [a-zA-Z0-9_-]{{1,64}}: {custom_id!r}"
            )
        if custom_id in custom_ids:
            return _anthropic_error(400, "invalid_request_error", f"duplicate custom_id: {custom_id!r}")
        custom_ids.append(custom_id)
        params.pop("model", None)
        params.pop("stream", None)
        params.setdefault("anthropic_version", "bedrock-2023-05-31")
        records.append(json.dumps({"recordId": custom_id, "modelInput": params}, separators=(",", ":")))

    refused = _billing_preflight((deployment.get("model_info") or {}).get("id"))
    if refused is not None:
        # Refuse BEFORE staging input or creating the job — no cleanup, no
        # unbillable-job window.
        return refused

    ctx = _bedrock_context(deployment)
    input_key = f"{_S3_INPUT_PREFIX}{uuid.uuid4().hex}.jsonl"
    s3_url = f"https://{ctx['bucket']}.s3.{ctx['region']}.amazonaws.com/{input_key}"
    upload = await _aws_call("PUT", s3_url, "\n".join(records) + "\n", "s3", ctx["params"], ctx["region"])
    if upload.status_code != 200:
        verbose_proxy_logger.error("bedrock msgbatch S3 staging failed: %s %s", upload.status_code, upload.text[:500])
        return _anthropic_error(502, "api_error", "failed to stage batch input")

    job_request = {
        "modelId": ctx["model_id"],
        "jobName": f"anthropic-msgbatch-{uuid.uuid4().hex[:12]}",
        "roleArn": ctx["batch_role_arn"],
        "inputDataConfig": {"s3InputDataConfig": {"s3Uri": f"s3://{ctx['bucket']}/{input_key}"}},
        "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": f"s3://{ctx['bucket']}/{_S3_OUTPUT_PREFIX}"}},
        "timeoutDurationInHours": 24,
    }
    create = await _aws_call(
        "POST",
        f"https://bedrock.{ctx['region']}.amazonaws.com/model-invocation-job",
        json.dumps(job_request),
        "bedrock",
        ctx["params"],
        ctx["region"],
    )
    if create.status_code != 200:
        verbose_proxy_logger.error("bedrock msgbatch create failed: %s %s", create.status_code, create.text[:500])
        return _anthropic_error(502, "api_error", "failed to create Bedrock batch job")
    job_arn = create.json()["jobArn"]
    job_id = job_arn.rsplit("/", 1)[1]
    client_batch_id = f"{BEDROCK_MSGBATCH_PREFIX}{job_id}_{_owner_tag(user_api_key_dict)}"

    recorded = await _record_batch_for_billing(
        provider_batch_id=job_arn,
        router_model_id=(deployment.get("model_info") or {}).get("id"),
        client_batch_id=client_batch_id,
        # The client-facing name (not the internal "-batch" alias): the stash
        # drives the spend log's model attribution; Bedrock pricing itself
        # comes from the deployment's explicit *_cost_per_token_batches keys.
        model_name=str(single_model),
        total_records=len(custom_ids),
        user_api_key_dict=user_api_key_dict,
    )
    if not recorded and _billing_required():
        # Never run an unbillable job: stop it and VERIFY the stop took — an
        # ignored failure leaves a running, untracked job (codex P1).
        stop = await _aws_call("POST", _job_url(job_id, ctx, "/stop"), None, "bedrock", ctx["params"], ctx["region"])
        stop_confirmed = stop.status_code in (200, 202)
        if not stop_confirmed:
            try:
                job, _ = await _get_bedrock_job(client_batch_id)
                stop_confirmed = job.get("status") in ({"Stopping"} | _ENDED_STATUSES)
            except Exception:
                pass
        log = verbose_proxy_logger.error if stop_confirmed else verbose_proxy_logger.critical
        log(
            "messages/batches billing: refused Bedrock batch %s (billing row not stored; stop status %s, confirmed=%s)%s",
            client_batch_id,
            stop.status_code,
            stop_confirmed,
            "" if stop_confirmed else " — job may be RUNNING WITHOUT COST TRACKING, reconcile manually",
        )
        return _anthropic_error(503, "api_error", "batch accounting is unavailable; the batch was not accepted")

    now = datetime.datetime.now(datetime.timezone.utc)
    return JSONResponse(
        status_code=200,
        content={
            "id": client_batch_id,
            "type": "message_batch",
            "processing_status": "in_progress",
            "request_counts": {
                "processing": len(custom_ids),
                "succeeded": 0,
                "errored": 0,
                "canceled": 0,
                "expired": 0,
            },
            "created_at": _rfc3339(now),
            "expires_at": _rfc3339(now + datetime.timedelta(hours=24)),
            "ended_at": None,
            "archived_at": None,
            "cancel_initiated_at": None,
            "results_url": None,
        },
    )


def _results_base_url(request: Request) -> str:
    # Prefer the operator-configured public URL: X-Forwarded-* / Host are
    # caller-controlled, and results_url is followed by SDK clients WITH their
    # gateway key attached — trusting those headers would let a caller point
    # other users' keys at an attacker origin. PROXY_BASE_URL is LiteLLM's
    # existing public-URL convention; request.url is the untrusted fallback
    # for bare deployments (no forwarded headers consulted).
    configured = os.getenv("PROXY_BASE_URL")
    if configured:
        return configured.rstrip("/")
    return f"{request.url.scheme}://{request.url.netloc}"


async def _get_bedrock_job(batch_id: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    deployment = _any_bedrock_batch_deployment()
    if deployment is None:
        raise HTTPException(
            status_code=404,
            detail={"type": "error", "error": {"type": "not_found_error", "message": f"batch {batch_id} not found"}},
        )
    ctx = _bedrock_context(deployment)
    job_id, _owner8 = _split_bedrock_batch_id(batch_id)
    response = await _aws_call("GET", _job_url(job_id, ctx), None, "bedrock", ctx["params"], ctx["region"])
    if response.status_code == 404 or response.status_code == 400:
        raise HTTPException(
            status_code=404,
            detail={"type": "error", "error": {"type": "not_found_error", "message": f"batch {batch_id} not found"}},
        )
    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail={"type": "error", "error": {"type": "api_error", "message": "Bedrock job lookup failed"}},
        )
    return response.json(), ctx


@router.get(
    "/v1/messages/batches/{batch_id}",
    tags=["[beta] Anthropic `/v1/messages/batches`"],
    dependencies=[Depends(user_api_key_auth)],
)
async def retrieve_message_batch(
    batch_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    if not batch_id.startswith(BEDROCK_MSGBATCH_PREFIX):
        await _check_upstream_batch_owner(batch_id, user_api_key_dict)
        return await _forward_upstream(request, "GET", f"/v1/messages/batches/{_url_quote(batch_id, safe='')}")
    await _check_bedrock_batch_owner(batch_id, user_api_key_dict)
    job, _ctx = await _get_bedrock_job(batch_id)
    return JSONResponse(status_code=200, content=_map_job_to_message_batch(job, batch_id, _results_base_url(request)))


@router.get(
    "/v1/messages/batches/{batch_id}/results",
    tags=["[beta] Anthropic `/v1/messages/batches`"],
    dependencies=[Depends(user_api_key_auth)],
)
async def message_batch_results(
    batch_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    if not batch_id.startswith(BEDROCK_MSGBATCH_PREFIX):
        await _check_upstream_batch_owner(batch_id, user_api_key_dict)
        return await _forward_upstream(request, "GET", f"/v1/messages/batches/{_url_quote(batch_id, safe='')}/results")

    await _check_bedrock_batch_owner(batch_id, user_api_key_dict)
    job, ctx = await _get_bedrock_job(batch_id)
    status = job.get("status")
    if status not in _ENDED_STATUSES:
        return _anthropic_error(
            404, "not_found_error", f"batch {batch_id} has not finished processing (status: {status})"
        )

    job_id, _owner8 = _split_bedrock_batch_id(batch_id)
    input_uri = job["inputDataConfig"]["s3InputDataConfig"]["s3Uri"]
    output_prefix = job["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"].removeprefix(f"s3://{ctx['bucket']}/")
    input_key = input_uri.removeprefix(f"s3://{ctx['bucket']}/")
    input_basename = input_key.rsplit("/", 1)[-1]
    output_key = f"{output_prefix}{job_id}/{input_basename}.out"

    async def _s3_get(key: str) -> Optional[str]:
        url = f"https://{ctx['bucket']}.s3.{ctx['region']}.amazonaws.com/{key}"
        response = await _aws_call("GET", url, None, "s3", ctx["params"], ctx["region"])
        return response.text if response.status_code == 200 else None

    output_text = await _s3_get(output_key)
    input_text = await _s3_get(input_key)

    # Fail loudly instead of streaming an empty/partial "success": if the job
    # processed records but the output object is unreadable, or the input
    # object (needed to reconstruct never-processed custom_ids) is gone while
    # records are missing, a 200 here would silently drop results.
    processed_any = int(job.get("successRecordCount") or 0) + int(job.get("errorRecordCount") or 0) > 0
    if output_text is None and processed_any:
        return _anthropic_error(502, "api_error", "batch output is not readable from S3")
    if input_text is None and status != "Completed":
        return _anthropic_error(502, "api_error", "batch input is not readable from S3")

    seen: Dict[str, Dict[str, Any]] = {}
    if output_text:
        for line in output_text.splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                return _anthropic_error(502, "api_error", "batch output contains malformed JSONL")
            record_id = record.get("recordId", "")
            if "modelOutput" in record:
                seen[record_id] = {"type": "succeeded", "message": record["modelOutput"]}
            elif "error" in record:
                seen[record_id] = _bedrock_error_to_anthropic(record["error"])

    # Records missing from the output: expired (ran out of time), canceled
    # (user stop), or errored (whole-job failure) depending on the job state.
    missing_type = {"Stopped": {"type": "canceled"}, "Expired": {"type": "expired"}}.get(
        str(status),
        {"type": "expired"}
        if status == "PartiallyCompleted"
        else {
            "type": "errored",
            "error": {
                "type": "error",
                "request_id": None,
                "error": {"type": "api_error", "message": str(job.get("message") or "batch job failed")},
            },
        },
    )
    all_ids: List[str] = []
    if input_text:
        for line in input_text.splitlines():
            if line.strip():
                all_ids.append(json.loads(line).get("recordId", ""))

    def _iter_lines():
        emitted = set()
        for record_id in all_ids or list(seen):
            result = seen.get(record_id)
            if result is None and status == "Completed":
                # Counter/output lag shouldn't happen on Completed; be explicit.
                result = {
                    "type": "errored",
                    "error": {
                        "type": "error",
                        "request_id": None,
                        "error": {"type": "api_error", "message": "result missing from batch output"},
                    },
                }
            yield json.dumps({"custom_id": record_id, "result": result or missing_type}, separators=(",", ":")) + "\n"
            emitted.add(record_id)
        for record_id, result in seen.items():
            if record_id not in emitted:
                yield json.dumps({"custom_id": record_id, "result": result}, separators=(",", ":")) + "\n"

    return StreamingResponse(_iter_lines(), media_type="application/x-jsonl")


@router.post(
    "/v1/messages/batches/{batch_id}/cancel",
    tags=["[beta] Anthropic `/v1/messages/batches`"],
    dependencies=[Depends(user_api_key_auth)],
)
async def cancel_message_batch(
    batch_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    if not batch_id.startswith(BEDROCK_MSGBATCH_PREFIX):
        await _check_upstream_batch_owner(batch_id, user_api_key_dict)
        return await _forward_upstream(request, "POST", f"/v1/messages/batches/{_url_quote(batch_id, safe='')}/cancel")
    await _check_bedrock_batch_owner(batch_id, user_api_key_dict)
    job, ctx = await _get_bedrock_job(batch_id)
    job_id, _owner8 = _split_bedrock_batch_id(batch_id)
    if job.get("status") not in _ENDED_STATUSES:
        stop = await _aws_call("POST", _job_url(job_id, ctx, "/stop"), None, "bedrock", ctx["params"], ctx["region"])
        job, ctx = await _get_bedrock_job(batch_id)
        if stop.status_code not in (200, 202) and job.get("status") not in {"Stopping", *_ENDED_STATUSES}:
            # Stop failed AND the refetched job doesn't independently confirm
            # a stop — surface the failure instead of fabricating "canceling".
            verbose_proxy_logger.error("bedrock msgbatch stop failed: %s %s", stop.status_code, stop.text[:300])
            return _anthropic_error(502, "api_error", "failed to cancel the Bedrock batch job")
    mapped = _map_job_to_message_batch(job, batch_id, _results_base_url(request))
    if mapped["processing_status"] != "ended":
        mapped["processing_status"] = "canceling"
    if mapped["cancel_initiated_at"] is None:
        mapped["cancel_initiated_at"] = _rfc3339(datetime.datetime.now(datetime.timezone.utc))
    return JSONResponse(status_code=200, content=mapped)


# ── Owner-scoped listing (ledger-derived) ────────────────────────────────────

# Non-admins can't be shown the upstream enumeration (shared workspace), so
# their listing is rebuilt from the billing ledger the create legs registered.
_LIST_DEFAULT_LIMIT = 20
_LIST_MAX_LIMIT = 100
# Take for the TEMPORARY legacy bridge (see _fetch_legacy_owner_batches): a
# per-owner, column-indexed (created_by / team_id) query over owner_key IS NULL
# message batches a pre-#335 writer left behind. The CheckBatchCost poller
# self-heals those NULLs within a poll cycle (owner_key backfill), so this is a
# small, transient read-time fallback with no scan and no cap.
_LEGACY_FALLBACK_CAP = 1000


def _row_file_object(row: object) -> dict[str, Any]:
    """The row's file_object as a dict (the write-path stores it JSON-encoded,
    occasionally double-encoded — tolerate two levels, same as
    CheckBatchCost._get_job_file_object and _row_attribution)."""
    file_object = getattr(row, "file_object", None)
    for _ in range(2):
        if not isinstance(file_object, str):
            break
        try:
            file_object = json.loads(file_object)
        except ValueError:
            return {}
    return file_object if isinstance(file_object, dict) else {}


def _client_batch_id(row: object) -> Optional[str]:
    """The CLIENT-facing message-batch id for a row THIS route created, or None
    if the row isn't ours (skip it).

    Must survive CheckBatchCost finalization, which REWRITES file_object from the
    provider response — overwriting `id` with the provider batch id / ARN and
    turning created_at into an epoch int (check_batch_cost.py::_finalized_file_object).

    Resolution order:
      1. `litellm_client_batch_id`, stashed at create and on the finalization
         preserve-list — authoritative, finalization-proof, and its presence is
         the POSITIVE marker that distinguishes an Anthropic message batch from
         an OpenAI-dialect /v1/batches Bedrock job (whose jobArn looks identical
         but never carries this key). This is the path for every row minted
         after this fix shipped.
      2. Back-compat for pre-marker rows: a raw msgbatch_* model_object_id is
         UNIQUE to this route's upstream leg (Anthropic message batches are
         created only here) and equals the client id — a column, so it survives
         finalization.
      3. A pre-finalization Bedrock row still carries the owner-tagged client id
         in file_object.id. A FINALIZED pre-marker Bedrock row has lost it, and a
         bare jobArn cannot be told apart from an OpenAI-dialect Bedrock job, so
         it is deliberately NOT surfaced — never leak a foreign-dialect job into
         the Anthropic listing (codex P1)."""
    file_object = _row_file_object(row)
    stashed = file_object.get("litellm_client_batch_id")
    if isinstance(stashed, str) and stashed.startswith("msgbatch_"):
        return stashed
    model_object_id = str(getattr(row, "model_object_id", "") or "")
    if model_object_id.startswith("msgbatch_"):
        return model_object_id
    file_object_id = file_object.get("id")
    if isinstance(file_object_id, str) and file_object_id.startswith(BEDROCK_MSGBATCH_PREFIX):
        return file_object_id
    return None


def _to_rfc3339(value: object) -> Optional[str]:
    """Normalize a stored timestamp to RFC3339. Accepts an RFC3339 string (the
    pre-finalization stash writes created_at that way) OR a Unix epoch int/float
    (LiteLLMBatch.created_at / .expires_at, written when CheckBatchCost finalizes
    the row). Returns None when absent/unusable so the caller can fall back —
    never silently substitutes now() for a real, finalized timestamp (that made
    completed batches list as newly-created and drift expires_at every request)."""
    if isinstance(value, str) and value:
        return value
    # bool is an int subclass — exclude it explicitly.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return _rfc3339(datetime.datetime.fromtimestamp(value, datetime.timezone.utc))
        except (ValueError, OverflowError, OSError):
            return None
    return None


def _list_expires_at(created_at: str) -> str:
    """created_at + 24h (both the Anthropic batch window and the Bedrock job
    timeout are 24h), falling back to created_at itself if the stored timestamp
    doesn't parse — expires_at must stay a valid RFC3339 string, never null."""
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            parsed = datetime.datetime.strptime(created_at, fmt).replace(tzinfo=datetime.timezone.utc)
            return _rfc3339(parsed + datetime.timedelta(hours=24))
        except ValueError:
            continue
    return created_at


def _render_ledger_batch(row: object, results_base_url: str, client_id: str) -> dict[str, Any]:
    """Render one ledger row as an Anthropic `message_batch`, exposing ONLY
    non-tenant-identifying fields — litellm_attribution and the row's
    created_by/team_id never leave the server. `client_id` is resolved by
    _client_batch_id (finalization-proof), so the returned id always works
    against GET /v1/messages/batches/{id}.

    processing_status comes from the poller-advanced COLUMN status: `ended`
    when it is in _FINALIZED_ROW_STATUSES (the same "done" definition the delete
    gate uses), else `in_progress` — including the transient `pricing` claim.
    This is a LEDGER view, not a live provider poll: a batch that finished at
    the provider but whose row the poller hasn't advanced yet still reads
    in_progress. RETRIEVE /{id} gives the live status; this listing exists to
    re-identify batch ids."""
    file_object = _row_file_object(row)
    ended = str(getattr(row, "status", "") or "") in _FINALIZED_ROW_STATUSES
    total = int(file_object.get("total_records") or 0)
    if ended:
        # The ledger doesn't record the terminal per-record split (succeeded /
        # errored / …) — RETRIEVE/{id} does. Report zeros rather than fabricate
        # a breakdown we don't have.
        counts = {"processing": 0, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}
    else:
        counts = {"processing": total, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}
    # created_at is an RFC3339 string pre-finalization and an epoch int after —
    # accept both; only invent now() when the value is genuinely absent.
    created_at = _to_rfc3339(file_object.get("created_at")) or _rfc3339(datetime.datetime.now(datetime.timezone.utc))
    # Prefer the finalized provider expires_at (epoch); otherwise derive +24h.
    # Both are stable across requests (the bug was deriving from a now()-stamped
    # created_at every call).
    expires_at = _to_rfc3339(file_object.get("expires_at")) or _list_expires_at(created_at)
    return {
        "id": client_id,
        "type": "message_batch",
        "processing_status": "ended" if ended else "in_progress",
        "request_counts": counts,
        "created_at": created_at,
        "expires_at": expires_at,
        # The ledger doesn't store a precise end timestamp; RETRIEVE/{id} does.
        "ended_at": None,
        "archived_at": None,
        "cancel_initiated_at": None,
        # Gateway base only — never api.anthropic.com — and only once ended, so
        # the client's gateway key (not the proxy's upstream credential) fetches
        # results through our own /results route (same rule as _forward_upstream).
        "results_url": f"{results_base_url}/v1/messages/batches/{client_id}/results" if ended else None,
    }


def _list_query_get(request: Request, key: str) -> Optional[str]:
    value = request.query_params.get(key)
    return value if value else None


def _list_limit(request: Request) -> int:
    """Anthropic `limit`: default 20, capped at 100, floored at 1. A malformed
    value falls back to the default rather than erroring (unknown/garbage query
    params must not fail the request)."""
    raw = request.query_params.get("limit")
    try:
        limit = int(raw) if raw is not None else _LIST_DEFAULT_LIMIT
    except (TypeError, ValueError):
        limit = _LIST_DEFAULT_LIMIT
    return max(1, min(limit, _LIST_MAX_LIMIT))


def _owner_or_predicate(user_api_key_dict: UserAPIKeyAuth) -> list[dict[str, Any]]:
    """The SQL OR-branch that matches rows this caller owns, mirroring
    _row_matches_caller. For rows THIS route created, the attribution stash and
    the columns agree (owner_key == attribution.user_api_key; team_id/created_by
    == the submitting key's team/user), so the three column predicates below
    reproduce all five _row_matches_caller conditions. A caller value that is
    None is dropped (never matches every-row-with-a-NULL-column)."""
    owner_or: list[dict[str, Any]] = []
    if user_api_key_dict.api_key:
        owner_or.append({"owner_key": user_api_key_dict.api_key})
    if user_api_key_dict.team_id:
        owner_or.append({"team_id": user_api_key_dict.team_id})
    if user_api_key_dict.user_id:
        owner_or.append({"created_by": user_api_key_dict.user_id})
    return owner_or


def _keyset_seek_predicate(seek: tuple[Any, Any], op: str) -> dict[str, Any]:
    """Strict keyset seek on (created_at, unified_object_id): rows past the
    cursor in the given direction. op is "lt" (older, forward/desc paging) or
    "gt" (newer, backward/asc paging). The unified_object_id tie-breaker is
    REQUIRED so rows sharing a created_at can't be straddled or duplicated."""
    seek_created_at, seek_unified_object_id = seek
    return {
        "OR": [
            {"created_at": {op: seek_created_at}},
            {
                "AND": [
                    {"created_at": seek_created_at},
                    {"unified_object_id": {op: seek_unified_object_id}},
                ]
            },
        ]
    }


async def _cursor_seek_point(cursor_id: str, user_api_key_dict: UserAPIKeyAuth) -> Optional[tuple[Any, Any]]:
    """Resolve a client-facing cursor id to its (created_at, unified_object_id)
    keyset point, scoped to the caller's ownership. Inverts the id the same way
    _get_billing_row does (Bedrock ids match on the jobArn suffix, upstream ids
    on the raw model_object_id). Returns None — an EMPTY page, matching the old
    'unknown cursor -> empty' semantics — when no such row exists OR the caller
    doesn't own it OR it isn't one of this route's message batches. Raises on a
    DB error so the caller's 503 wrapper fails closed."""
    from litellm.proxy.proxy_server import prisma_client

    where: dict[str, Any] = {}
    if cursor_id.startswith(BEDROCK_MSGBATCH_PREFIX):
        job_id, _owner8 = _split_bedrock_batch_id(cursor_id)
        where["model_object_id"] = {"endswith": f"/model-invocation-job/{job_id}"}
    else:
        where["model_object_id"] = cursor_id
    row = await prisma_client.db.litellm_managedobjecttable.find_first(where=where)
    if row is None or _client_batch_id(row) is None or not _row_matches_caller(row, user_api_key_dict):
        return None
    created_at = getattr(row, "created_at", None)
    unified_object_id = getattr(row, "unified_object_id", None)
    if created_at is None or unified_object_id is None:
        return None
    return (created_at, unified_object_id)


async def _fetch_legacy_owner_batches(
    user_api_key_dict: UserAPIKeyAuth,
    results_base_url: str,
    *,
    seek: Optional[tuple[Any, Any]],
    op: str,
    order_dir: str,
) -> list[tuple[Any, Any, dict[str, Any]]]:
    """TEMPORARY legacy bridge — delete once pre-migration / rolling-deploy rows
    have drained (TODO(owner_key-backfill); batches expire in 24h). The indexed
    primary query requires owner_key IS NOT NULL, so a message-batch row a pre-#335
    writer left with owner_key NULL would be missing until it is healed. The
    CheckBatchCost poller SELF-HEALS those NULLs each cycle (it derives owner_key
    from the attribution stash and backfills the column), so there is NO read-time
    scan and NO cap here — just one cheap INDEXED fallback query:

      Column-scoped (created_by / team_id): an old writer stamps created_by/team_id
      from the same key, so the column matches the caller's own NULL rows. This
      covers callers carrying those columns immediately (before the poller runs),
      and is merged into the keyset window with the primary rows.

    A key-hash-only caller (no user_id AND no team_id) has no column to match, so
    a NULL row of theirs is served by retrieve-by-id until the poller backfills its
    owner_key (within a poll cycle), at which point the indexed primary query picks
    it up — a brief, self-healing gap, by design, not a read-time scan."""
    if not (user_api_key_dict.user_id or user_api_key_dict.team_id):
        return []
    from litellm.proxy.proxy_server import prisma_client

    order = [{"created_at": order_dir}, {"unified_object_id": order_dir}]
    seek_and: list[dict[str, Any]] = [_keyset_seek_predicate(seek, op)] if seek is not None else []
    out: list[tuple[Any, Any, dict[str, Any]]] = []
    seen: set[Any] = set()

    legacy_or: list[dict[str, Any]] = []
    if user_api_key_dict.user_id:
        legacy_or.append({"created_by": user_api_key_dict.user_id})
    if user_api_key_dict.team_id:
        legacy_or.append({"team_id": user_api_key_dict.team_id})
    # Scoped to the single owner's columns, so a per-owner take cap is enough.
    where: dict[str, Any] = {"file_purpose": "batch", "owner_key": None, "AND": [{"OR": legacy_or}, *seek_and]}
    rows = await prisma_client.db.litellm_managedobjecttable.find_many(
        where=where, order=order, take=_LEGACY_FALLBACK_CAP
    )
    _collect_legacy_rows(rows, user_api_key_dict, results_base_url, out, seen)
    return out


def _collect_legacy_rows(
    rows: list[Any],
    user_api_key_dict: UserAPIKeyAuth,
    results_base_url: str,
    out: list[tuple[Any, Any, dict[str, Any]]],
    seen: set[Any],
) -> None:
    """Filter a legacy NULL-owner_key result set to this caller's message
    batches and append (created_at, unified_object_id, rendered) tuples, deduped
    by unified_object_id. _client_batch_id excludes OpenAI-dialect /v1/batches
    rows (no message-batch client id); _row_matches_caller re-checks ownership
    (columns AND the attribution key hash) for the exact pre-#335 semantics."""
    for row in rows:
        unified_object_id = getattr(row, "unified_object_id", None)
        if unified_object_id in seen:
            continue
        client_id = _client_batch_id(row)
        if client_id is None or not _row_matches_caller(row, user_api_key_dict):
            continue
        seen.add(unified_object_id)
        out.append(
            (
                getattr(row, "created_at", None),
                unified_object_id,
                _render_ledger_batch(row, results_base_url, client_id),
            )
        )


async def _fetch_owner_page(
    user_api_key_dict: UserAPIKeyAuth,
    owner_or: list[dict[str, Any]],
    results_base_url: str,
    *,
    limit: int,
    seek: Optional[tuple[Any, Any]],
    forward: bool,
) -> tuple[list[dict[str, Any]], bool]:
    """One keyset page of the caller's owned message batches, newest-first.

    forward=True pages toward OLDER rows (no cursor / after_id): seek `lt`,
    order desc. forward=False pages toward NEWER rows (before_id): seek `gt`,
    order asc (the `limit` rows immediately newer than the cursor), then the
    page is reversed back to desc. take = limit + |legacy| + 1 guarantees the
    bounded legacy rows can interleave without dropping a primary row from the
    window; the +1 detects has_more."""
    from litellm.proxy.proxy_server import prisma_client

    op = "lt" if forward else "gt"
    order_dir = "desc" if forward else "asc"
    legacy = await _fetch_legacy_owner_batches(
        user_api_key_dict, results_base_url, seek=seek, op=op, order_dir=order_dir
    )
    and_clauses: list[dict[str, Any]] = [{"OR": owner_or}]
    if seek is not None:
        and_clauses.append(_keyset_seek_predicate(seek, op))
    where: dict[str, Any] = {
        # owner_key IS NOT NULL is the message-batch route discriminator: it
        # excludes OpenAI-dialect /v1/batches rows (which never set it) even
        # when they share this owner's created_by/team_id.
        "file_purpose": "batch",
        "owner_key": {"not": None},
        "AND": and_clauses,
    }
    rows = await prisma_client.db.litellm_managedobjecttable.find_many(
        where=where,
        order=[{"created_at": order_dir}, {"unified_object_id": order_dir}],
        take=limit + len(legacy) + 1,
    )
    candidates: list[tuple[Any, Any, dict[str, Any]]] = []
    for row in rows:
        client_id = _client_batch_id(row)
        if client_id is None:
            continue  # defensive: owner_key IS NOT NULL should imply one of ours
        candidates.append(
            (
                getattr(row, "created_at", None),
                getattr(row, "unified_object_id", None),
                _render_ledger_batch(row, results_base_url, client_id),
            )
        )
    candidates.extend(legacy)  # disjoint from primary (owner_key NULL vs NOT NULL)
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=forward)
    has_more = len(candidates) > limit
    page = [rendered for _ca, _uid, rendered in candidates[:limit]]
    if not forward:
        page.reverse()  # backward fetch is ascending; present newest-first
    return page, has_more


async def _list_owner_scoped_batches(request: Request, user_api_key_dict: UserAPIKeyAuth) -> JSONResponse:
    """Build the caller's owner-scoped message-batch listing from the ledger
    with a single INDEXED keyset query (owner_key / team_id / created_by are
    all indexed), replacing the old chunked full-table scan + Python owner
    filter. Ownership is expressed in SQL via _owner_or_predicate, which
    reproduces _row_matches_caller for every row this route created (the
    backfilled owner_key column carries the key hash the old filter matched in
    the attribution JSON). owner_key IS NOT NULL is the message-batch route
    discriminator (OpenAI-dialect /v1/batches rows never set it), replacing the
    old _client_batch_id-not-None post-filter."""
    limit = _list_limit(request)
    after_id = _list_query_get(request, "after_id")
    before_id = _list_query_get(request, "before_id")
    if after_id is not None and before_id is not None:
        # Anthropic treats the two cursors as mutually exclusive (passing both is
        # a client error upstream); reject rather than silently prefer one.
        return _anthropic_error(
            400,
            "invalid_request_error",
            "after_id and before_id cannot be used together; pass at most one",
        )
    base = _results_base_url(request)

    owner_or = _owner_or_predicate(user_api_key_dict)
    if not owner_or:
        # A caller with no key hash, user, or team owns nothing (mirrors
        # _row_matches_caller returning False for every row) — empty page.
        return _empty_batch_list_response()

    cursor_id = after_id or before_id
    forward = before_id is None  # after_id / no-cursor page toward OLDER rows

    try:
        seek: Optional[tuple[Any, Any]] = None
        if cursor_id is not None:
            seek = await _cursor_seek_point(cursor_id, user_api_key_dict)
            if seek is None:
                return _empty_batch_list_response()  # unknown cursor -> empty page
        page, has_more = await _fetch_owner_page(
            user_api_key_dict, owner_or, base, limit=limit, seek=seek, forward=forward
        )
    except Exception:  # noqa: BLE001
        # Fail CLOSED on ANY DB read failure: a partial/empty result must never
        # masquerade as "you have no batches", so we catch broadly and 503 (a
        # raise would 500 raw). Narrowing to prisma.errors.PrismaError is the
        # codebase convention, but importing prisma here is not viable — it is
        # generated only at proxy build time and is absent from the unit-test
        # environment, so a hot-path import would ImportError every listing test.
        verbose_proxy_logger.exception("messages/batches: owner-scoped list query failed")
        return _anthropic_error(503, "api_error", "batch listing is temporarily unavailable; retry shortly")

    return JSONResponse(
        status_code=200,
        content={
            "data": page,
            "has_more": has_more,
            "first_id": page[0]["id"] if page else None,
            "last_id": page[-1]["id"] if page else None,
        },
    )


def _empty_batch_list_response() -> JSONResponse:
    return JSONResponse(
        status_code=200,
        content={"data": [], "has_more": False, "first_id": None, "last_id": None},
    )


@router.get(
    "/v1/messages/batches",
    tags=["[beta] Anthropic `/v1/messages/batches`"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_message_batches(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    # The upstream forward enumerates the proxy's SHARED Anthropic workspace, so
    # exposing it to a virtual key leaks (and enables reading) OTHER tenants'
    # batches — the reason the old handler hard-403'd every non-admin (codex P1,
    # fork PR #131). But that left an owner unable to re-identify their own lost
    # batch ids. So: admins still get the verbatim upstream forward (full
    # workspace view); non-admins get an OWNER-SCOPED listing rebuilt from the
    # billing ledger — the cross-tenant enumeration stays closed, and Bedrock
    # jobs (never in the upstream listing) now surface for their owner.
    if _is_proxy_admin(user_api_key_dict):
        query = f"?{request.url.query}" if request.url.query else ""
        return await _forward_upstream(request, "GET", f"/v1/messages/batches{query}")

    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        # No DB => no per-owner ledger to scope against. Falling back to the
        # upstream forward here would reopen the exact cross-tenant enumeration
        # the admin gate closed, so refuse instead (admins above keep the
        # shared-workspace view even without a DB).
        return _anthropic_error(
            403,
            "permission_error",
            "listing batches requires a database-backed proxy (no per-key batch ledger is "
            "available); fetch a known batch with GET /v1/messages/batches/{batch_id} instead",
        )
    return await _list_owner_scoped_batches(request, user_api_key_dict)


@router.delete(
    "/v1/messages/batches/{batch_id}",
    tags=["[beta] Anthropic `/v1/messages/batches`"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_message_batch(
    batch_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    if not batch_id.startswith(BEDROCK_MSGBATCH_PREFIX):
        await _check_upstream_batch_owner(batch_id, user_api_key_dict)
        blocked = await _unbilled_delete_block(batch_id, user_api_key_dict)
        if blocked is not None:
            return blocked
        return await _forward_upstream(request, "DELETE", f"/v1/messages/batches/{_url_quote(batch_id, safe='')}")
    await _check_bedrock_batch_owner(batch_id, user_api_key_dict)
    blocked = await _unbilled_delete_block(batch_id, user_api_key_dict)
    if blocked is not None:
        # Deleting would remove the S3 output CheckBatchCost prices from —
        # never before the spend is written (codex P1).
        return blocked
    job, ctx = await _get_bedrock_job(batch_id)
    if job.get("status") not in _ENDED_STATUSES:
        # Same precondition as Anthropic: an in-progress batch must be
        # canceled before it can be deleted.
        return _anthropic_error(400, "invalid_request_error", "batch must finish processing before it can be deleted")
    # Best-effort removal of the S3 artifacts (Bedrock has no job-delete API;
    # the job record itself ages out server-side).
    job_id, _owner8 = _split_bedrock_batch_id(batch_id)
    input_uri = job["inputDataConfig"]["s3InputDataConfig"]["s3Uri"]
    input_key = input_uri.removeprefix(f"s3://{ctx['bucket']}/")
    output_prefix = job["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"].removeprefix(f"s3://{ctx['bucket']}/")
    output_key = f"{output_prefix}{job_id}/{input_key.rsplit('/', 1)[-1]}.out"
    for key in (input_key, output_key, f"{output_prefix}{job_id}/manifest.json.out"):
        url = f"https://{ctx['bucket']}.s3.{ctx['region']}.amazonaws.com/{key}"
        response = await _aws_call("DELETE", url, None, "s3", ctx["params"], ctx["region"])
        if response.status_code not in (200, 204):
            verbose_proxy_logger.warning("bedrock msgbatch artifact delete failed: %s %s", key, response.status_code)
    return JSONResponse(status_code=200, content={"id": batch_id, "type": "message_batch_deleted"})
