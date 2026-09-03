#!/usr/bin/env python3
"""
Verify that every patch recorded in .github/fork-patches.txt is still present.

This is the "daily-patch-verify" the manifest's own header has described since
it was created, and that (until this script) nothing implemented -- see
rayward-internal/llm-gateway-infra#694. A -X theirs upstream sync can silently
drop a fork-only hunk with zero merge conflict (git's 3-way merge just loses a
change that never conflicts textually with the surrounding diff); this script
is the only thing standing between that and the patch being gone with no
signal, exactly like the digest-pin drift that broke the image build from
2026-08-30 to 2026-09-02 (rayward-external/litellm#239) -- the manifest row
for that patch was PRESENT and asserted the pin was present, and it was: the
literal digest had just gone stale relative to the ARG default it shadows,
and nothing compared the two. Presence-only is exactly the failure mode this
script must not repeat, so read the "Row kinds" note below before assuming a
plain grep is enough.

Manifest format (.github/fork-patches.txt):

    <file>[ / <file> ...] | <pattern> | <description>

  - <file> is repo-relative. Multiple files separated by " / " means the
    pattern must be found in at least one of them (used by rows that
    describe the same fact landing in several budget files at once).
  - <pattern> is matched two ways, either of which counts as present:
      1. as a literal substring (handles the common case: most patterns are
         literal source-code fragments, and several contain unescaped
         regex metacharacters -- '(', '.', '[' -- that were written with
         literal intent, e.g. row `py.detach(f)` in gil.rs, where an
         unescaped '(' turns "detach(f)" into a *regex group* that no
         longer requires the literal parens. Trying the literal substring
         first means a mis-escaped-as-regex pattern still gets credit
         for the literal code it names.)
      2. as a Python regex (MULTILINE, deliberately NOT DOTALL -- see
         pattern_present), for the handful of rows that deliberately use
         regex features (`^`/`$` anchors, `\\(`/`\\)`/`\\.` escapes for
         literal parens/dots, an explicit `[\\s\\S]` where a row really
         must span lines). Invalid regexes (unbalanced parens/brackets that
         were meant as literal code, e.g. `expect(...).toEqual([])`) fall
         back to the literal check only.
    A pattern of exactly `N/A` marks a row that has no local, greppable
    artifact to check -- see "Row kinds" below -- and is always skipped.

Row kinds (why this is not a blanket "grep every row" script):

  1. Current-assertion rows (the majority): the pattern must be found in
     the named file(s) *right now*. A miss here is exactly the "patch
     reverted to vanilla" failure this script exists to catch, and it
     fails the run.

  2. Ratchet-log rows: every row whose file(s) all end in `-budget.json`
     (ruff-strict-budget.json, type-discipline-budget.json,
     basedpyright-code-budget.json, test-quality-budget.json). These
     files hold live numeric ceilings that a separate, pre-existing gate
     (scripts/ruff_strict_gate.py and siblings, wired into
     test-linting.yml) ratchets on every PR; fork-patches.txt rows for
     them are a chronological changelog of past hand-raised ceilings, not
     a claim that stayed true. Measured directly against this tree
     (2026-09-02): the manifest's three most recent "C901" rows read
     319 -> 318 -> 314 in that order, while ruff-strict-budget.json's
     actual current C901 limit is 311 -- lower than every one of them,
     because `make lint-budget-update` ratchets a rule down whenever the
     branch's own count improves, silently, with no manifest row. A
     literal-match requirement here would be permanently and correctly
     red on ~30% of the manifest (56 of 198 rows) for values that were
     never wrong. These rows are still parsed and their file(s) are
     confirmed to exist (catches a renamed/deleted budget file), but
     their pattern is not matched. Their own currency is enforced by the
     ratchet gate they describe, not by this script.

  3. Historical / superseded rows, marked `N/A`: narrative incident
     write-ups (a "file" field that is actually an event title, e.g.
     "2026-08-05 sync merge corruption (12 files, ...)") and individual
     rows superseded by a later row recording the same fact's current
     value (e.g. a cryptography floor bumped from >=49.0.0 to >=50.0.0
     got its own later row; the earlier one is kept for audit trail and
     marked N/A rather than deleted). Skipped, loudly, in the summary.

Phase 2 -- digest pins must not be STALE, not merely present:

  Presence is necessary but not sufficient for one whole class of row. The
  fork replaces upstream's `FROM $LITELLM_BUILD_IMAGE` indirection with a
  literal `FROM <image>@sha256:<digest>` so OpenSSF Scorecard's
  PinnedDependencies check passes. A literal pin is not a variable
  reference, so when an upstream sync bumps the `ARG ..._IMAGE` default the
  literal `FROM` stays behind and nothing compares the two. The row keeps
  asserting "a pin is present" -- and it is; it is just the WRONG pin.

  That is what made the image unbuildable 2026-08-30 -> 2026-09-02: the ARG
  moved to a wolfi-base with glibc 2.44 for its new python3.13 while the
  literal FROM stayed on glibc 2.43 (rayward-external/litellm#239). It is
  not a one-off: this phase's first run found the SAME drift live and
  undetected on the `ui-builder` stage of Dockerfile,
  docker/Dockerfile.database and docker/Dockerfile.non_root -- upstream
  moved `ARG UI_BUILD_IMAGE` to node:24.19-alpine3.24 on 2026-08-04 (commit
  487074f602, "move the Admin UI toolchain to Node 24"; the dashboard's own
  package.json now declares `engines: {"node": ">=24.14.1"}`) while the
  literal FROM stayed on node:20.18-alpine3.20.

  So phase 2 compares, for every Dockerfile the manifest names, each
  literal `FROM <ref>@sha256:<digest>` against the `ARG` default it
  shadows, matching the FULL reference (name, tag AND digest -- the node
  drift above changed the tag too, so a digest-only comparison keyed on an
  exact repo:tag string does not even pair the two lines up). Stage names
  map to the ARG upstream's un-patched `FROM $VAR` would have used; any
  other stage falls back to matching on the image name with its tag
  stripped. REQUIRED_PINNED_DOCKERFILES makes the phase non-vacuous at file
  level, and EXPECTED_LITERAL_PIN_STAGES does it at STAGE level: every
  (file, stage) the fork deliberately pins must still BE a literal pin, so
  reverting a single stage to `FROM $VAR` fails instead of quietly dropping
  out of the comparison.

Usage:
    python3 scripts/verify_fork_patches.py

Exit status: 0 if every current-assertion row's pattern was found, every
referenced file exists, and every literal digest pin still equals the ARG
default it tracks; 1 otherwise, with every failure listed.
"""

from __future__ import annotations

import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST_PATH = os.path.join(REPO_ROOT, ".github", "fork-patches.txt")

# --- Phase 2: literal digest pin vs ARG default ------------------------------

# Dockerfiles that MUST be covered by the pin check. Listed explicitly so the
# phase cannot go vacuous if a manifest row is dropped, a file is renamed, or
# the FROM/ARG regexes stop matching. These are the five that carried the
# 2026-08-30 drift (rayward-internal/llm-gateway-infra#694);
# docker/Dockerfile.non_root is checked too when the manifest names it, but is
# not required (its builder/runtime stages are still vanilla `FROM $VAR`).
# Rows currently marked `pattern = N/A` (never checked). Pinned so that
# silencing a LIVE row by flipping its pattern to N/A -- the cheapest possible
# way to make this script green without fixing anything -- shows up as a diff
# to this number and has to be argued for in review.
EXPECTED_NA_ROWS = 15

REQUIRED_PINNED_DOCKERFILES = (
    "Dockerfile",
    "backend/Dockerfile",
    "gateway/Dockerfile",
    "migrations/Dockerfile",
    "docker/Dockerfile.database",
)

# Stage name -> the ARG that upstream's un-patched `FROM $VAR` would have used.
# `docker/Dockerfile.database`'s runtime stage is the reason this map exists at
# stage granularity: only its runtime stage is literally pinned, and in the
# 2026-08-30 incident it had drifted to a THIRD digest that would have built
# fine and died at run time on the glibc mismatch.
STAGE_TO_ARG = {
    "builder": "LITELLM_BUILD_IMAGE",
    "runtime": "LITELLM_RUNTIME_IMAGE",
    "ui-builder": "UI_BUILD_IMAGE",
    "uvbin": "UV_IMAGE",
}

# Every (file, stage) that MUST be a literal `FROM <image>@sha256:` pin.
#
# Anti-vacuity has to be per-STAGE, not per-file. A file-level "did we compare
# anything here?" guard is satisfied by any one surviving pin, so reverting a
# SINGLE stage back to `FROM $LITELLM_RUNTIME_IMAGE` used to pass green: the
# reverted line simply stopped being a literal pin, so nothing compared it, and
# the file still had three other pins to vouch for it. Measured on this tree
# before this map existed: reverting docker/Dockerfile.database's `runtime`
# stage -- exactly the stage rayward-internal/llm-gateway-infra#694 calls out,
# the one that would have built clean and died at run time -- exited 0.
#
# 18 pins across 6 files. Note this is a SUPERSET of the "all 9 literal
# digests" the manifest's 2026-09-02 quarantine row counts: that 9 is the
# wolfi-base subset #239 re-synced for the glibc break (builder+runtime in the
# four wolfi files, plus Dockerfile.database's runtime). The other 9 are the
# six `uvbin` (ghcr.io/astral-sh/uv) and three `ui-builder` (node) pins, which
# drift and revert exactly the same way -- the node ui-builder pins are the
# ones that were actually found stale.
EXPECTED_LITERAL_PIN_STAGES = {
    "Dockerfile": ("uvbin", "ui-builder", "builder", "runtime"),
    "backend/Dockerfile": ("uvbin", "builder", "runtime"),
    "gateway/Dockerfile": ("uvbin", "builder", "runtime"),
    "migrations/Dockerfile": ("uvbin", "builder", "runtime"),
    # builder is still vanilla `FROM $LITELLM_BUILD_IMAGE` here.
    "docker/Dockerfile.database": ("uvbin", "ui-builder", "runtime"),
    # builder and runtime are still vanilla `FROM $VAR` here.
    "docker/Dockerfile.non_root": ("uvbin", "ui-builder"),
}

_ARG_PIN_RE = re.compile(
    r"^ARG\s+(?P<name>[A-Z0-9_]+_IMAGE)\s*=\s*(?P<ref>\S+@sha256:[0-9a-f]{64})\s*$",
    re.MULTILINE,
)
_FROM_PIN_RE = re.compile(
    r"^FROM\s+(?:--\S+\s+)*(?P<ref>\S+@sha256:[0-9a-f]{64})"
    r"(?:\s+AS\s+(?P<stage>\S+))?\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def image_name(ref: str) -> str:
    """`node:24.19-alpine3.24@sha256:ab...` -> `node`; registry paths kept whole."""
    ref = ref.split("@", 1)[0]
    head, sep, tail = ref.rpartition(":")
    # A ':' is only a tag separator when what follows has no '/' (else it is a
    # registry port, e.g. `localhost:5000/img`).
    if sep and "/" not in tail:
        return head
    return ref


def _is_dockerfile_path(path: str) -> bool:
    base = os.path.basename(path)
    return base == "Dockerfile" or base.startswith("Dockerfile.")


def dockerfiles_named_by_manifest(rows: list[Row]) -> list[str]:
    """Every Dockerfile path the manifest records a patch for, deduplicated.

    The manifest is the single source of truth for WHICH files carry the pin
    patch; REQUIRED_PINNED_DOCKERFILES only asserts the answer still covers
    the five that must never go unchecked.
    """
    seen: dict[str, None] = {}
    for row in rows:
        for f in row.files:
            if _is_dockerfile_path(f):
                seen.setdefault(f, None)
    return list(seen)


def dockerfile_pin_failures(rows: list[Row], repo_root: str = REPO_ROOT) -> list[str]:
    """Every literal FROM digest that no longer equals the ARG default it tracks.

    `repo_root` is injectable so the negative tests can run the real rule
    against a synthetic tree instead of mutating the working copy.
    """
    # (file_key, message). The key is structured rather than recovered by
    # substring-matching the rendered message: `rel_path in message` treated
    # "Dockerfile" as present in "docker/Dockerfile.database: ...", so the root
    # Dockerfile's own vacuity finding was suppressed by any failure naming a
    # DIFFERENT Dockerfile.
    found: list[tuple[str | None, str]] = []
    named = dockerfiles_named_by_manifest(rows)

    missing_from_manifest = [f for f in REQUIRED_PINNED_DOCKERFILES if f not in named]
    if missing_from_manifest:
        found.append(
            (None, f"manifest names no patch row for {missing_from_manifest}, so the pin "
            f"check would not cover them. Either the row was dropped by an upstream "
            f"sync (restore it) or the file was renamed (update the row and "
            f"REQUIRED_PINNED_DOCKERFILES).")
        )

    pinned_stages_seen: dict[str, set[str]] = {}

    for rel_path in sorted(named):
        full_path = os.path.join(repo_root, rel_path)
        if not os.path.isfile(full_path):
            if rel_path in REQUIRED_PINNED_DOCKERFILES or rel_path in EXPECTED_LITERAL_PIN_STAGES:
                found.append(
                    (rel_path, f"{rel_path}: required pinned Dockerfile does not exist")
                )
            continue
        with open(full_path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()

        args = {m.group("name"): m.group("ref") for m in _ARG_PIN_RE.finditer(text)}
        by_name: dict[str, dict[str, str]] = {}
        for name, ref in args.items():
            by_name.setdefault(image_name(ref), {})[name] = ref

        pinned_stages_seen[rel_path] = set()

        # Build- and runtime-ARGs for the SAME image must not disagree: they are
        # the same base image in every current Dockerfile, so a split means a
        # bump was applied to only one of them.
        for name, refs in sorted(by_name.items()):
            if len(set(refs.values())) > 1:
                rendered = ", ".join(f"{n}={r}" for n, r in sorted(refs.items()))
                found.append(
                    (rel_path, f"{rel_path}: ARG defaults for `{name}` disagree ({rendered}). "
                    f"A base-image bump appears to have been applied to only some of "
                    f"them, so whichever literal FROM tracks the un-bumped ARG is now "
                    f"pinned to a different image than the rest of the build.")
                )

        for m in _FROM_PIN_RE.finditer(text):
            ref = m.group("ref")
            stage = (m.group("stage") or "").lower()
            expected_arg = STAGE_TO_ARG.get(stage)
            pinned_stages_seen[rel_path].add(stage)

            if expected_arg and expected_arg in args:
                if ref != args[expected_arg]:
                    found.append(
                        (rel_path, f"{rel_path}: stage `{stage}` pins\n"
                        f"        {ref}\n"
                        f"    but {expected_arg} defaults to\n"
                        f"        {args[expected_arg]}\n"
                        f"    The literal FROM pin has DRIFTED from the ARG default it "
                        f"shadows. A literal pin is not a variable reference, so an "
                        f"upstream ARG bump leaves it behind with nothing comparing the "
                        f"two -- the exact failure that made this image unbuildable "
                        f"2026-08-30 -> 2026-09-02. Set the FROM to the ARG's value; do "
                        f"NOT change the ARG, and do NOT revert the pin to "
                        f"`FROM ${expected_arg}` (see .github/fork-patches.txt).")
                    )
                continue

            if expected_arg:
                found.append(
                    (rel_path, f"{rel_path}: stage `{stage}` is literally pinned but the file "
                    f"declares no `ARG {expected_arg}=...@sha256:...` to compare it "
                    f"against. The ARG was renamed or dropped, which leaves this pin "
                    f"tracking nothing.")
                )
                continue

            candidates = by_name.get(image_name(ref), {})
            if not candidates:
                # An image this file pins with no ARG counterpart at all: nothing
                # to drift against, so not a finding.
                continue
            if ref not in set(candidates.values()):
                rendered = ", ".join(f"{n}={r}" for n, r in sorted(candidates.items()))
                found.append(
                    (rel_path, f"{rel_path}: stage `{stage or '<unnamed>'}` pins\n"
                    f"        {ref}\n"
                    f"    which matches none of this file's ARG defaults for that "
                    f"image ({rendered}).\n"
                    f"    Set the FROM to the ARG's value. Do not revert the literal "
                    f"pin -- it is a deliberate fork patch.")
                )

    # Per-STAGE anti-vacuity. Every stage the fork deliberately pins must still
    # BE a literal pin. A per-file guard cannot see a single reverted stage:
    # the reverted line just stops matching _FROM_PIN_RE, so nothing compares
    # it, while the file's other pins keep the file looking covered.
    files_with_findings = {key for key, _ in found if key is not None}
    for rel_path, expected_stages in sorted(EXPECTED_LITERAL_PIN_STAGES.items()):
        if rel_path in files_with_findings and rel_path not in pinned_stages_seen:
            continue
        seen = pinned_stages_seen.get(rel_path)
        if seen is None:
            if rel_path not in files_with_findings:
                found.append(
                    (
                        rel_path,
                        f"{rel_path}: expected literal digest pins for stages "
                        f"{list(expected_stages)} but the file was never scanned -- "
                        f"it is not named by any .github/fork-patches.txt row, or it "
                        f"no longer exists.",
                    )
                )
            continue
        missing = [st for st in expected_stages if st not in seen]
        if missing:
            found.append(
                (
                    rel_path,
                    f"{rel_path}: stage(s) {missing} are no longer a literal "
                    f"`FROM <image>@sha256:...` pin (literal pins still present: "
                    f"{sorted(seen)}).\n"
                    f"    An upstream sync reverted that stage to the vanilla "
                    f"`FROM $VAR` form, or the pin was deleted. This is NOT caught by "
                    f"the file's other pins: a reverted stage simply stops being "
                    f"compared, which is why anti-vacuity is asserted per STAGE. "
                    f"Re-apply the literal pin (set it to the stage's ARG default); "
                    f"do not delete the row or this entry to make the check pass.",
                )
            )

    return [msg for _, msg in found]


class Row:
    __slots__ = ("line_no", "raw_file_field", "pattern", "description")

    def __init__(self, line_no: int, raw_file_field: str, pattern: str, description: str):
        self.line_no = line_no
        self.raw_file_field = raw_file_field
        self.pattern = pattern
        self.description = description

    @property
    def files(self) -> list[str]:
        return [p.strip() for p in self.raw_file_field.split(" / ")]

    @property
    def is_ratchet_log(self) -> bool:
        return all(f.endswith("-budget.json") for f in self.files)

    @property
    def is_historical(self) -> bool:
        return self.pattern.strip() == "N/A"


def parse_manifest(path: str) -> list[Row]:
    rows: list[Row] = []
    with open(path, encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split(" | ", 2)
            if len(parts) != 3:
                raise ValueError(
                    f"{path}:{line_no}: expected 3 ' | '-separated fields "
                    f"(<file> | <pattern> | <description>), got {len(parts)}: "
                    f"{line[:120]!r}"
                )
            file_field, pattern, description = (p.strip() for p in parts)
            if not file_field or not pattern:
                raise ValueError(f"{path}:{line_no}: empty file or pattern field: {line[:120]!r}")
            rows.append(Row(line_no, file_field, pattern, description))
    return rows


def pattern_present(pattern: str, content: str) -> bool:
    """True if `pattern` proves present in `content`, by either match strategy.

    MULTILINE but deliberately NOT DOTALL. Under DOTALL a `.` crosses newlines,
    so a row like `FROM.*sha256:` was satisfied by ANY `FROM` anywhere in the
    file followed by `sha256:` anywhere later -- i.e. one surviving pinned stage
    vouched for every other stage in the same file, and a per-stage revert to
    `FROM $LITELLM_RUNTIME_IMAGE` passed green. Measured on this tree: reverting
    docker/Dockerfile.database's runtime stage (the one #694 calls out as the
    silent-failure stage) exited 0 before this change.

    Only two rows in the manifest ever relied on DOTALL (the image-scan.yml
    `<job>-image ... --require-hashes` pairs); both were rewritten to tempered
    patterns that cannot cross into a neighbouring job. A row that genuinely
    needs to span lines should say so explicitly with `[\\s\\S]` or `\\n`
    rather than leaning on a global flag that silently widens every other row.
    """
    if pattern in content:
        return True
    try:
        rx = re.compile(pattern, re.MULTILINE)
    except re.error:
        return False
    return rx.search(content) is not None


def main() -> int:
    try:
        rows = parse_manifest(MANIFEST_PATH)
    except ValueError as e:
        print(f"::error::{e}")
        return 1

    if not rows:
        print(f"::error::{MANIFEST_PATH} yielded zero patch rows -- parser is broken or the file is empty.")
        return 1

    failures: list[str] = []
    missing_files: list[str] = []
    historical_skipped = 0
    ratchet_skipped = 0
    checked = 0

    for row in rows:
        if row.is_historical:
            historical_skipped += 1
            continue

        if row.is_ratchet_log:
            ratchet_skipped += 1
            for f in row.files:
                if not os.path.isfile(os.path.join(REPO_ROOT, f)):
                    missing_files.append(
                        f"line {row.line_no}: ratchet-log file missing: {f}"
                    )
            continue

        checked += 1
        found = False
        any_file_exists = False
        for f in row.files:
            full_path = os.path.join(REPO_ROOT, f)
            if not os.path.isfile(full_path):
                continue
            any_file_exists = True
            with open(full_path, encoding="utf-8", errors="replace") as fh:
                content = fh.read()
            if pattern_present(row.pattern, content):
                found = True
                break

        if not any_file_exists:
            missing_files.append(
                f"line {row.line_no}: none of {row.files} exist "
                f"(patch: {row.description[:100]!r}...)"
            )
            continue

        if not found:
            failures.append(
                f"line {row.line_no}: pattern not found in {row.files}\n"
                f"    pattern: {row.pattern!r}\n"
                f"    patch:   {row.description[:160]!r}..."
            )

    pin_failures = dockerfile_pin_failures(rows)

    print(f"fork-patches.txt: {len(rows)} rows total")
    print(f"  {checked} current-assertion rows checked")
    print(f"  {ratchet_skipped} ratchet-log rows skipped (see script docstring, kind 2)")
    print(f"  {historical_skipped} historical/superseded rows skipped (pattern=N/A, kind 3)")
    print(
        f"  {len(dockerfiles_named_by_manifest(rows))} manifest-named Dockerfiles "
        f"checked for STALE digest pins (phase 2)"
    )
    print()

    ok = True

    if historical_skipped > EXPECTED_NA_ROWS:
        ok = False
        print(
            f"N/A ROW COUNT ROSE: {historical_skipped} rows are marked "
            f"`pattern = N/A` (never checked), but EXPECTED_NA_ROWS is "
            f"{EXPECTED_NA_ROWS}. Flipping a live row to N/A silences it without "
            f"fixing anything. If the new N/A row is genuinely narrative or "
            f"superseded, raise EXPECTED_NA_ROWS in the same commit so the change "
            f"is visible in review."
        )
        print()

    if missing_files:
        ok = False
        print(f"MISSING FILES ({len(missing_files)}) -- a row references a path that no longer exists:")
        for m in missing_files:
            print(f"  - {m}")
        print()

    if failures:
        ok = False
        print(f"REVERTED OR DRIFTED PATCHES ({len(failures)}):")
        for fmsg in failures:
            print(f"  - {fmsg}")
        print()
        print(
            "A recorded fork patch's pattern is no longer present in its file. "
            "Either the patch was silently dropped by an upstream sync (re-apply "
            "it) or the surrounding code changed shape and the manifest's pattern "
            "is stale (update .github/fork-patches.txt to match). Do not delete "
            "the row to make this pass."
        )
        print()

    if pin_failures:
        ok = False
        print(f"STALE DIGEST PINS ({len(pin_failures)}):")
        for pmsg in pin_failures:
            print(f"  - {pmsg}")
        print()
        print(
            "A literal FROM digest pin no longer equals the ARG default it shadows. "
            "The pin is PRESENT but WRONG, which a presence-only manifest check "
            "cannot see -- that is the whole reason this phase exists "
            "(rayward-internal/llm-gateway-infra#694)."
        )

    if ok:
        print(f"OK: all {checked} current-assertion patches are present.")
        print("OK: every literal digest pin still equals the ARG default it tracks.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
