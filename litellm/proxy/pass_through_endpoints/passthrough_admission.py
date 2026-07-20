"""Admission control for pass-through endpoints.

Pass-through forwards a client request to an upstream provider using the
proxy's OWN credentials. If the response cannot be priced, the proxy bills the
upstream account and records `spend = 0` against the caller's key. That is
worse than a plain outage: budgets stop binding on that route, and any
reconciliation that splits a provider invoice across keys by gateway-computed
cost will silently redistribute the unpriced key's spend onto keys that priced
correctly.

Cost tracking on pass-through is an allow-list, not a guarantee: a handler must
recognise the provider AND the specific route AND the streaming mode. Routes
outside that allow-list return `$0` with no error.

This module refuses such requests **before the upstream call**. Checking after
the fact is useless — the money is already spent.

Enforcement is opt-in and fail-closed once enabled:

    general_settings:
      passthrough_require_cost_tracking: true
      passthrough_capabilities:
        - provider: anthropic
          methods: [POST]
          path: /v1/messages
          model_source: body            # read `model` from the request body
        - provider: bedrock
          methods: [POST]
          path: /model/{model_id}/converse
          model_source: path:model_id   # read it from the path placeholder

With `passthrough_require_cost_tracking: true` and no matching capability, the
request is rejected. Set it to false (the default) to preserve today's
behaviour.

Scope, stated honestly: this proves a costing path is *registered* and that the
model resolves to an explicit price entry. It cannot prove the upstream will
return usable usage — that is what a live smoke assertion and per-request
zero-cost alerting are for.
"""

import re
from collections.abc import Mapping
from typing import Any

from litellm._logging import verbose_proxy_logger

REQUIRE_COST_TRACKING_SETTING = "passthrough_require_cost_tracking"
CAPABILITIES_SETTING = "passthrough_capabilities"

# `{name}` matches exactly one path segment, so `/model/{id}/converse` cannot
# swallow `/model/a/b/converse`. Keeping placeholders segment-bound is what
# stops a template from widening into the subtree it was meant to exclude.
_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class PassthroughAdmissionError(Exception):
    """Raised when a pass-through request has no registered costing path."""

    def __init__(self, message: str, status_code: int = 403):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def _is_explicitly_true(value: Any) -> bool:
    """True only for a real boolean True or an explicit truthy scalar.

    Deliberately strict. An arbitrary object (a Mock, a config stub, anything
    whose `.get()` returns another object) must NOT count as "enabled" — that
    would turn admission control on by accident and reject every pass-through
    request with a 500.
    """
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    if isinstance(value, int) and not isinstance(value, bool):
        return value == 1
    return False


def _template_to_regex(path_template: str) -> re.Pattern:
    parts: list[str] = []
    last = 0
    for match in _PLACEHOLDER_RE.finditer(path_template):
        parts.append(re.escape(path_template[last : match.start()]))
        parts.append(f"(?P<{match.group(1)}>[^/]+)")
        last = match.end()
    parts.append(re.escape(path_template[last:]))
    return re.compile("^" + "".join(parts) + "$")


def _normalize_path(path: str) -> str:
    """Collapse duplicate slashes and drop a single trailing slash.

    Only cosmetic normalization belongs here. Percent-decoding deliberately does
    NOT: decoding `%2F` into `/` would let a caller synthesise a path that
    matches a narrower template than the one the upstream will actually route.
    """
    if not path:
        return "/"
    collapsed = re.sub(r"/{2,}", "/", path)
    if len(collapsed) > 1 and collapsed.endswith("/"):
        collapsed = collapsed[:-1]
    return collapsed or "/"


def _model_is_priced(model: str | None) -> bool:
    """True when `model` resolves to an explicit entry in the price map.

    A fallback or default price is treated as unpriced on purpose: a wrong
    non-zero cost is harder to detect than a zero one, because nothing looks
    broken.
    """
    if not model:
        return False

    import litellm

    if model in litellm.model_cost:
        return True
    # Providers commonly return a bare id where the map is prefixed
    # (`fireworks_ai/accounts/...`), or the reverse. Accept a prefixed form only
    # if it exists explicitly.
    if "/" in model and model.split("/", 1)[1] in litellm.model_cost:
        return True
    return any(
        key.endswith("/" + model)
        for key in litellm.model_cost  # type: ignore[union-attr]
    )


def _extract_model(
    capability: dict[str, Any],
    path_match: re.Match | None,
    request_body: dict | None,
) -> str | None:
    source = str(capability.get("model_source") or "body")
    if source.startswith("path:"):
        placeholder = source.split(":", 1)[1]
        if path_match is None:
            return None
        try:
            return path_match.group(placeholder)
        except (IndexError, KeyError):
            # Capability declares a placeholder its own path template does not
            # define — a config error, not a client error. Treat as unpriced so
            # it fails closed rather than forwarding unmetered.
            return None
    if not isinstance(request_body, dict):
        return None
    model = request_body.get("model")
    return str(model) if model else None


def find_matching_capability(
    capabilities: list[dict[str, Any]],
    provider: str | None,
    method: str,
    path: str,
) -> tuple[dict[str, Any] | None, re.Match | None]:
    normalized = _normalize_path(path)
    upper_method = (method or "").upper()
    for capability in capabilities:
        if not isinstance(capability, dict):
            continue
        cap_provider = capability.get("provider")
        if cap_provider and provider and str(cap_provider) != str(provider):
            continue
        methods = capability.get("methods") or []
        if methods and upper_method not in {str(m).upper() for m in methods}:
            continue
        template = str(capability.get("path") or "")
        if not template:
            continue
        match = _template_to_regex(_normalize_path(template)).match(normalized)
        if match:
            return capability, match
    return None, None


def enforce_passthrough_admission(
    general_settings: dict | None,
    provider: str | None,
    method: str,
    path: str,
    request_body: dict | None,
) -> None:
    """Raise PassthroughAdmissionError if this request has no costing path.

    No-op unless `passthrough_require_cost_tracking` is true.
    """
    # Enforcement must be turned on EXPLICITLY. Never infer it from
    # truthiness: `general_settings` is not guaranteed to be a plain dict, and
    # an object whose `.get()` returns another object would otherwise switch
    # enforcement on by accident and reject every pass-through request.
    if not isinstance(general_settings, Mapping):
        return
    if not _is_explicitly_true(general_settings.get(REQUIRE_COST_TRACKING_SETTING, False)):
        return

    capabilities = general_settings.get(CAPABILITIES_SETTING) or []
    if not isinstance(capabilities, list):
        raise PassthroughAdmissionError(f"{CAPABILITIES_SETTING} must be a list of capability objects", status_code=500)

    capability, path_match = find_matching_capability(capabilities, provider, method, path)
    if capability is None:
        verbose_proxy_logger.warning(
            "pass-through admission denied: no registered capability for provider=%s %s %s",
            provider,
            method,
            path,
        )
        raise PassthroughAdmissionError(
            f"Pass-through endpoint '{method} {path}' is not a registered capability"
            f"{f' for provider {provider}' if provider else ''}. "
            "It would bill the upstream provider without recording cost against your key. "
            f"Register it under general_settings.{CAPABILITIES_SETTING} once its cost tracking is verified."
        )

    # `require_priced_model` defaults to True: a capability is registered
    # precisely because we intend to price it.
    if capability.get("require_priced_model", True):
        model = _extract_model(capability, path_match, request_body)
        if not _model_is_priced(model):
            verbose_proxy_logger.warning(
                "pass-through admission denied: model %r has no explicit price entry (%s %s)",
                model,
                method,
                path,
            )
            raise PassthroughAdmissionError(
                f"Model '{model}' has no explicit price entry, so this pass-through request "
                "would be recorded at $0 while still billing the upstream provider. "
                "Add the model to the price map, or set require_priced_model: false on the "
                "capability if it is genuinely non-billable."
            )
