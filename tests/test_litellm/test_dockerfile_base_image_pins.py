"""
Static checks that literal base-image digest pins agree with the ARG defaults
in the same Dockerfile.

Upstream (BerriAI/litellm) writes `FROM $LITELLM_BUILD_IMAGE`. This fork
replaces that indirection with a literal `FROM ...@sha256:<digest>` so
OpenSSF Scorecard's PinnedDependencies check passes (see
`.github/fork-patches.txt`). That patch is deliberate and must stay.

The hazard the patch creates is drift: an upstream sync bumps the
`ARG LITELLM_BUILD_IMAGE` / `ARG LITELLM_RUNTIME_IMAGE` default, but the
literal `FROM` lines are not variable references so nothing updates them and
nothing compares them. The two silently diverge.

That is exactly what broke the image build on 2026-08-30. Upstream bumped the
ARG defaults to a wolfi-base carrying glibc 2.44 and switched the builder to
`apk add python-3.13` / `uv sync --python python3.13`, while the literal
`FROM` lines stayed on a May 2026 wolfi-base carrying glibc 2.43:

    ImportError: /usr/lib/libm.so.6: version `GLIBC_2.44' not found
        (required by .../math.cpython-313-x86_64-linux-gnu.so)

These tests fail on divergence so the next ARG bump cannot land half-applied.
"""

import os
import re

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Registries whose pins we compare. Keyed on the image path so a bare tag bump
# (e.g. a different repo entirely) does not trip the comparison.
_PINNED_IMAGE_REPOS = (
    "cgr.dev/chainguard/wolfi-base",
    "ghcr.io/astral-sh/uv",
)

_ARG_RE = re.compile(
    r"^ARG\s+(?P<name>[A-Z0-9_]+_IMAGE)\s*=\s*(?P<repo>\S+?)@sha256:(?P<digest>[0-9a-f]{64})\s*$",
    re.MULTILINE,
)
_FROM_RE = re.compile(
    r"^FROM\s+(?:--\S+\s+)*(?P<repo>\S+?)@sha256:(?P<digest>[0-9a-f]{64})"
    r"(?:\s+AS\s+(?P<stage>\S+))?\s*$",
    re.MULTILINE,
)

# Stage name -> the ARG that upstream's un-patched `FROM $VAR` would have used.
_STAGE_TO_ARG = {
    "builder": "LITELLM_BUILD_IMAGE",
    "runtime": "LITELLM_RUNTIME_IMAGE",
}

# Dockerfiles that carry the fork's literal-pin patch and are expected to have
# both an ARG default and at least one literal FROM to compare it against.
# Listed explicitly so this suite cannot silently go vacuous if a file is
# renamed or the discovery walk stops matching.
PATCHED_DOCKERFILES = (
    "Dockerfile",
    "backend/Dockerfile",
    "gateway/Dockerfile",
    "migrations/Dockerfile",
    "docker/Dockerfile.database",
)


def _discover_dockerfiles() -> list[str]:
    """Repo-relative paths of every Dockerfile, excluding vendored trees."""
    skip_dirs = {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        "site-packages",
    }
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for filename in filenames:
            if filename == "Dockerfile" or filename.startswith("Dockerfile."):
                if filename.endswith((".md", ".txt", ".dockerignore")):
                    continue
                found.append(
                    os.path.relpath(os.path.join(dirpath, filename), REPO_ROOT)
                )
    return sorted(found)


def _read(rel_path: str) -> str:
    with open(os.path.join(REPO_ROOT, rel_path), "r", encoding="utf-8") as f:
        return f.read()


def _arg_pins(text: str) -> dict[str, tuple[str, str]]:
    """ARG name -> (repo, digest), for pinned images we track."""
    return {
        m.group("name"): (m.group("repo"), m.group("digest"))
        for m in _ARG_RE.finditer(text)
        if m.group("repo") in _PINNED_IMAGE_REPOS
    }


def _from_pins(text: str) -> list[tuple[str, str, str | None]]:
    """(repo, digest, stage) for each literal FROM of a tracked image."""
    return [
        (m.group("repo"), m.group("digest"), m.group("stage"))
        for m in _FROM_RE.finditer(text)
        if m.group("repo") in _PINNED_IMAGE_REPOS
    ]


DOCKERFILES = _discover_dockerfiles()


def test_patched_dockerfiles_are_discovered():
    """Guard against this suite silently testing nothing."""
    missing = [p for p in PATCHED_DOCKERFILES if p not in DOCKERFILES]
    assert not missing, (
        f"Dockerfiles carrying the fork's literal-digest-pin patch were not "
        f"discovered: {missing}. Either they were renamed (update "
        f"PATCHED_DOCKERFILES and .github/fork-patches.txt) or _discover_dockerfiles() "
        f"stopped matching them. Discovered: {DOCKERFILES}"
    )


@pytest.mark.parametrize("dockerfile", PATCHED_DOCKERFILES)
def test_patched_dockerfile_has_both_arg_and_literal_from(dockerfile: str):
    """Each patched file must actually have something to compare."""
    text = _read(dockerfile)

    args = _arg_pins(text)
    assert args, (
        f"{dockerfile} has no `ARG ..._IMAGE=<repo>@sha256:...` default. The "
        "drift check compares literal FROM pins against these; without one there "
        "is nothing to compare and the fork's pin can drift undetected."
    )

    froms = _from_pins(text)
    assert froms, (
        f"{dockerfile} has no literal `FROM <repo>@sha256:...` line. This fork "
        "pins base images by literal digest for Scorecard PinnedDependencies "
        "(.github/fork-patches.txt); an upstream sync appears to have reverted it "
        "to `FROM $LITELLM_BUILD_IMAGE`. Re-apply the pin."
    )


@pytest.mark.parametrize("dockerfile", DOCKERFILES)
def test_literal_from_digests_match_arg_defaults(dockerfile: str):
    """Every literal pinned FROM digest must equal an ARG default in the same file.

    This is the check that would have caught the 2026-08-30 build break.
    """
    text = _read(dockerfile)

    froms = _from_pins(text)
    if not froms:
        pytest.skip(f"{dockerfile} pins no tracked base image by literal digest")

    args = _arg_pins(text)
    if not args:
        pytest.skip(f"{dockerfile} declares no pinned ARG default to compare against")

    for repo, digest, stage in froms:
        # Digests to compare against: ARG defaults for the same image repo.
        same_repo = {
            name: d for name, (r, d) in args.items() if r == repo
        }
        if not same_repo:
            continue

        # Precise check: a `builder`/`runtime` stage must match the specific ARG
        # upstream's `FROM $VAR` form would have used.
        expected_arg = _STAGE_TO_ARG.get(stage or "")
        if expected_arg and expected_arg in same_repo:
            assert digest == same_repo[expected_arg], (
                f"{dockerfile}: stage `{stage}` pins {repo}@sha256:{digest} but "
                f"{expected_arg} defaults to sha256:{same_repo[expected_arg]}.\n"
                f"The literal FROM pin has drifted from the ARG default. Upstream "
                f"bumps the ARG; the literal FROM is not a variable reference so "
                f"nothing updates it. Set the FROM digest to the ARG's value "
                f"(do NOT change the ARG, and do NOT revert the pin to "
                f"`FROM ${expected_arg}` -- see .github/fork-patches.txt).\n"
                f"This exact drift broke the image build on 2026-08-30: the ARG "
                f"moved to a wolfi-base with glibc 2.44 for python3.13 while the "
                f"FROM stayed on glibc 2.43, and `uv sync --python python3.13` "
                f"died with \"version `GLIBC_2.44' not found\"."
            )
            continue

        # Looser check for any other stage: must match *some* ARG default.
        assert digest in set(same_repo.values()), (
            f"{dockerfile}: stage `{stage or '<unnamed>'}` pins "
            f"{repo}@sha256:{digest}, which matches none of this file's ARG "
            f"defaults for that image "
            f"({', '.join(f'{n}=sha256:{d}' for n, d in sorted(same_repo.items()))}).\n"
            f"The literal FROM pin has drifted from the ARG default; set it to the "
            f"ARG's value. Do not revert the literal pin -- it is a deliberate "
            f"fork patch (.github/fork-patches.txt)."
        )


@pytest.mark.parametrize("dockerfile", DOCKERFILES)
def test_pinned_arg_defaults_are_self_consistent(dockerfile: str):
    """Build and runtime ARGs for the same image repo must not disagree.

    They are the same base image in every current Dockerfile; a split would mean
    a bump was applied to only one of them.
    """
    args = _arg_pins(_read(dockerfile))
    if not args:
        pytest.skip(f"{dockerfile} declares no pinned ARG defaults")

    by_repo: dict[str, dict[str, str]] = {}
    for name, (repo, digest) in args.items():
        by_repo.setdefault(repo, {})[name] = digest

    for repo, names in by_repo.items():
        distinct = set(names.values())
        assert len(distinct) == 1, (
            f"{dockerfile}: ARG defaults for {repo} disagree "
            f"({', '.join(f'{n}=sha256:{d}' for n, d in sorted(names.items()))}). "
            f"A base-image bump appears to have been applied to only some of them."
        )
