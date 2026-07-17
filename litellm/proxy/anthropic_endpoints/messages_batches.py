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

Known MVP gaps (documented in the fork PR): no per-request spend logging on
this route (Bedrock spend reconciles via the provider-level cost feed), and
list/delete are upstream-only (Bedrock jobs don't surface there).
"""

import datetime
import hashlib
import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from litellm._logging import verbose_proxy_logger
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body

router = APIRouter()

BEDROCK_MSGBATCH_PREFIX = "msgbatch_bedrock_"
_ANTHROPIC_VERSION_DEFAULT = "2023-06-01"
_CUSTOM_ID_PATTERN = re.compile(r"\A[a-zA-Z0-9_-]{1,64}\Z")
_S3_INPUT_PREFIX = "anthropic-messages-batches/input/"
_S3_OUTPUT_PREFIX = "anthropic-messages-batches/output/"
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
    async with httpx.AsyncClient(timeout=120.0) as client:
        return await client.request(method, url, headers=headers, content=payload)


# ── Upstream (api.anthropic.com) forwarding ──────────────────────────────────


def _upstream_base() -> str:
    return os.getenv("ANTHROPIC_API_BASE") or os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"


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
    return {
        "x-api-key": api_key,
        "anthropic-version": request.headers.get("anthropic-version", _ANTHROPIC_VERSION_DEFAULT),
        "content-type": "application/json",
    }


async def _forward_upstream(
    request: Request,
    method: str,
    path: str,
    body: Optional[bytes] = None,
) -> Response:
    url = f"{_upstream_base()}{path}"
    async with httpx.AsyncClient(timeout=600.0) as client:
        upstream = await client.request(method, url, headers=_upstream_headers(request), content=body)
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
    return Response(content=content, status_code=upstream.status_code, media_type=media_type)


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
        processing_status = "canceling"
        counts.update(processing=remainder, succeeded=succeeded, errored=errored)
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
        processing_status = "in_progress"
        counts.update(processing=remainder if processed else total, succeeded=succeeded, errored=errored)

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
        "cancel_initiated_at": _rfc3339(datetime.datetime.now(datetime.timezone.utc)) if status in ("Stopping",) else None,
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
    single_model = models.pop() if len(models) == 1 else None
    # Bedrock enforces a fixed 100-record minimum per invocation job, so small
    # batches take the Anthropic-native leg even for Bedrock-batch-backed
    # models (same models, same 50% batch rate — just a different backend).
    deployment = (
        _find_bedrock_batch_deployment(single_model)
        if single_model and len(requests_list) >= 100
        else None
    )
    if deployment is None:
        # Mixed-model batches, sub-100-record batches, and models without a
        # Bedrock batch backend run on the Anthropic API upstream (which
        # supports every claude model).
        return await _forward_upstream(request, "POST", "/v1/messages/batches", json.dumps(body).encode())

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
    job_id = create.json()["jobArn"].rsplit("/", 1)[1]

    now = datetime.datetime.now(datetime.timezone.utc)
    return JSONResponse(
        status_code=200,
        content={
            "id": f"{BEDROCK_MSGBATCH_PREFIX}{job_id}",
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
    # Honor the LB-forwarded host so results_url points back at the gateway.
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.headers.get("host", request.url.netloc))
    return f"{proto}://{host}"


async def _get_bedrock_job(batch_id: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    deployment = _any_bedrock_batch_deployment()
    if deployment is None:
        raise HTTPException(
            status_code=404,
            detail={"type": "error", "error": {"type": "not_found_error", "message": f"batch {batch_id} not found"}},
        )
    ctx = _bedrock_context(deployment)
    job_id = batch_id.removeprefix(BEDROCK_MSGBATCH_PREFIX)
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
        return await _forward_upstream(request, "GET", f"/v1/messages/batches/{batch_id}")
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
        return await _forward_upstream(request, "GET", f"/v1/messages/batches/{batch_id}/results")

    job, ctx = await _get_bedrock_job(batch_id)
    status = job.get("status")
    if status not in _ENDED_STATUSES:
        return _anthropic_error(
            404, "not_found_error", f"batch {batch_id} has not finished processing (status: {status})"
        )

    job_id = batch_id.removeprefix(BEDROCK_MSGBATCH_PREFIX)
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

    seen: Dict[str, Dict[str, Any]] = {}
    if output_text:
        for line in output_text.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
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
        return await _forward_upstream(request, "POST", f"/v1/messages/batches/{batch_id}/cancel")
    job, ctx = await _get_bedrock_job(batch_id)
    job_id = batch_id.removeprefix(BEDROCK_MSGBATCH_PREFIX)
    if job.get("status") not in _ENDED_STATUSES:
        stop = await _aws_call(
            "POST", _job_url(job_id, ctx, "/stop"), None, "bedrock", ctx["params"], ctx["region"]
        )
        if stop.status_code not in (200, 202):
            verbose_proxy_logger.error("bedrock msgbatch stop failed: %s %s", stop.status_code, stop.text[:300])
        job, ctx = await _get_bedrock_job(batch_id)
    mapped = _map_job_to_message_batch(job, batch_id, _results_base_url(request))
    if mapped["processing_status"] != "ended":
        mapped["processing_status"] = "canceling"
    if mapped["cancel_initiated_at"] is None:
        mapped["cancel_initiated_at"] = _rfc3339(datetime.datetime.now(datetime.timezone.utc))
    return JSONResponse(status_code=200, content=mapped)


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
    # Upstream-only: Bedrock jobs are not merged into the listing (documented).
    query = f"?{request.url.query}" if request.url.query else ""
    return await _forward_upstream(request, "GET", f"/v1/messages/batches{query}")
