"""
Static checks that literal base-image digest pins agree with the ARG defaults
in the same Dockerfile.

Upstream (BerriAI/litellm) writes `FROM $LITELLM_BUILD_IMAGE`. This fork
replaces that indirection with a literal `FROM ...@sha256:<digest>` so
OpenSSF Scorecard's PinnedDependencies check passes (see
`.github/fork-patches.txt`). That patch is deliberate and must stay.

The hazard the patch creates is drift: an upstream sync bumps the
`ARG LITELLM_BUILD_IMAGE` / `ARG LITELLM_RUNTIME_IMAGE` / `ARG UI_BUILD_IMAGE`
default, but the literal `FROM` lines are not variable references so nothing
updates them and nothing compares them. The two silently diverge.

That is exactly what broke the image build on 2026-08-30. Upstream bumped the
ARG defaults to a wolfi-base carrying glibc 2.44 and switched the builder to
`apk add python-3.13` / `uv sync --python python3.13`, while the literal
`FROM` lines stayed on a May 2026 wolfi-base carrying glibc 2.43:

    ImportError: /usr/lib/libm.so.6: version `GLIBC_2.44' not found
        (required by .../math.cpython-313-x86_64-linux-gnu.so)

The RULE ITSELF now lives in `scripts/verify_fork_patches.py`, because it must
also run on the daily `fork-patch-verify.yml` schedule and after every upstream
sync -- not only when someone happens to run this suite
(rayward-internal/llm-gateway-infra#694). This module is the pytest entry point
to that same implementation, so the two can never disagree about what "pinned"
means. The original private copy of the rule here was keyed on an allowlist of
two image repos (wolfi-base, astral-sh/uv) and compared digests only, which is
why it reported green for a month while the `ui-builder` stage of three
Dockerfiles sat pinned to node:20.18-alpine3.20 after upstream moved
`ARG UI_BUILD_IMAGE` to node:24.19-alpine3.24 (commit 487074f602, 2026-08-04).
"""

import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_VERIFIER_PATH = os.path.join(REPO_ROOT, "scripts", "verify_fork_patches.py")


def _load_verifier():
    """Import scripts/verify_fork_patches.py by path (scripts/ is not a package).

    A failure here is a real finding, not a harness problem: it means the
    fork-only verifier script this fork's patch policy depends on is gone or
    unimportable, which is precisely the "an upstream sync wiped a fork-only
    file" case `.github/fork-patches.txt` records rows for.
    """
    spec = importlib.util.spec_from_file_location("_fork_patch_verifier", _VERIFIER_PATH)
    assert spec is not None and spec.loader is not None, (
        f"cannot load {_VERIFIER_PATH}; scripts/verify_fork_patches.py is the "
        "implementation of the fork-patch and digest-pin checks (see "
        ".github/fork-patches.txt). If an upstream sync removed it, restore it."
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verifier = _load_verifier()

REQUIRED = verifier.REQUIRED_PINNED_DOCKERFILES
ROWS = verifier.parse_manifest(verifier.MANIFEST_PATH)
NAMED_DOCKERFILES = verifier.dockerfiles_named_by_manifest(ROWS)
PIN_FAILURES = verifier.dockerfile_pin_failures(ROWS)


def test_manifest_names_every_required_dockerfile():
    """Guard against the pin check silently covering nothing.

    `.github/fork-patches.txt` is the single source of truth for which files
    carry the literal-pin patch; this asserts that answer still includes every
    Dockerfile that must never go unchecked.
    """
    missing = [p for p in REQUIRED if p not in NAMED_DOCKERFILES]
    assert not missing, (
        f"no .github/fork-patches.txt row names {missing}, so the digest-pin "
        f"check would not cover them. Either an upstream sync dropped the row "
        f"(restore it) or the file was renamed. Manifest currently names: "
        f"{NAMED_DOCKERFILES}"
    )


@pytest.mark.parametrize("dockerfile", REQUIRED)
def test_required_dockerfile_exists(dockerfile: str):
    assert os.path.isfile(os.path.join(REPO_ROOT, dockerfile)), (
        f"{dockerfile} is recorded as carrying the fork's literal-digest-pin "
        f"patch but does not exist."
    )


@pytest.mark.parametrize("dockerfile", REQUIRED)
def test_no_stale_literal_from_pins(dockerfile: str):
    """Every literal pinned FROM must equal the ARG default it shadows.

    This is the check that would have caught the 2026-08-30 build break.
    """
    mine = [f for f in PIN_FAILURES if f.startswith(f"{dockerfile}:")]
    assert not mine, "\n".join(mine)


def test_no_stale_literal_from_pins_anywhere():
    """Catch-all, including files outside REQUIRED (e.g. docker/Dockerfile.non_root)."""
    assert not PIN_FAILURES, "\n".join(PIN_FAILURES)


def test_pin_check_is_not_vacuous():
    """The rule must actually be comparing something.

    `dockerfile_pin_failures` reports a failure when a required Dockerfile
    yields zero compared pins, so a green run here plus the assertions above
    means real FROM/ARG pairs were compared -- not that the parser matched
    nothing.
    """
    assert NAMED_DOCKERFILES, (
        ".github/fork-patches.txt names no Dockerfile at all -- the manifest "
        "rows for the literal-pin patch are gone."
    )
    assert verifier.image_name(
        "node:24.19-alpine3.24@sha256:" + "d" * 64
    ) == "node", "image_name() no longer strips tags; pin pairing would break silently"
