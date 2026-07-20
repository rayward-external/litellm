from typing import Optional, Tuple
from urllib.parse import urlparse

from fastapi import Request

# Hostnames that route to OpenAI-compatible APIs.
#
# `openai.com` is OpenAI proper — kept as a suffix (rather than the exact
# `api.openai.com`) so regional/alternate API subdomains stay covered. The two
# Azure domains below are *shared by
# every Azure Cognitive Service* (Speech, Vision, Language, ...), not just Azure
# OpenAI: `openai.azure.com` is the classic Azure OpenAI domain, while
# `cognitiveservices.azure.com` is used by newer "Azure AI Foundry" /
# Cognitive Services-hosted Azure OpenAI deployments. Because the hostname alone
# cannot tell Azure OpenAI apart from the other Cognitive Services on those
# domains, requests there must additionally carry an OpenAI-style path segment.
OPENAI_HOSTNAMES = ("openai.com",)
AZURE_OPENAI_HOSTNAMES = ("openai.azure.com", "cognitiveservices.azure.com")
# Path markers that identify an Azure request as Azure OpenAI rather than Speech
# / Vision / Language / ... `/openai/` is the native Azure OpenAI path prefix;
# `/v1/` is the OpenAI-v1 surface used by LiteLLM's pass-through routing. Other
# Cognitive Services use service-named prefixes and versions like `/v3.1/`,
# `/v1.0/`, so they do not collide with these markers.
AZURE_OPENAI_PATH_MARKERS = ("/openai/", "/v1/")


def hostname_matches(hostname: str, suffixes: Tuple[str, ...]) -> bool:
    """True if hostname equals one of `suffixes` or is a subdomain of it.

    Uses suffix matching (not a bare substring test) so look-alikes such as
    `cognitiveservices.azure.com.attacker.example` are not accepted.
    """
    return any(
        hostname == suffix or hostname.endswith("." + suffix) for suffix in suffixes
    )


def is_openai_compatible_url(url_route: Optional[str]) -> bool:
    """True if the URL targets an OpenAI-compatible API surface.

    For the shared Azure Cognitive Services domains we additionally require an
    OpenAI-style path segment (`/openai/` or `/v1/`) so non-OpenAI Azure services
    (Speech, Vision, Language, ...) on the same domain are not misclassified as
    OpenAI routes.
    """
    if not url_route:
        return False
    parsed_url = urlparse(url_route)
    hostname = parsed_url.hostname
    if not hostname:
        return False
    if hostname_matches(hostname, OPENAI_HOSTNAMES):
        return True
    if hostname_matches(hostname, AZURE_OPENAI_HOSTNAMES):
        return any(marker in parsed_url.path for marker in AZURE_OPENAI_PATH_MARKERS)
    return False


def get_litellm_virtual_key(request: Request) -> str:
    """
    Extract and format API key from request headers.
    Prioritizes x-litellm-api-key over Authorization header.


    Vertex JS SDK uses `Authorization` header, we use `x-litellm-api-key` to pass litellm virtual key

    """
    litellm_api_key = request.headers.get("x-litellm-api-key")
    if litellm_api_key:
        return f"Bearer {litellm_api_key}"
    return request.headers.get("Authorization", "")
