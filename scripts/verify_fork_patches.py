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
      2. as a Python regex (MULTILINE | DOTALL), for the handful of rows
         that deliberately use regex features (`.*` spanning a job's
         style across lines, `\\(`/`\\)`/`\\.` escapes for literal
         parens/dots). Invalid regexes (unbalanced parens/brackets that
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

Usage:
    python3 scripts/verify_fork_patches.py

Exit status: 0 if every current-assertion row's pattern was found (and
every referenced file exists); 1 otherwise, with every failure listed.
"""

from __future__ import annotations

import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST_PATH = os.path.join(REPO_ROOT, ".github", "fork-patches.txt")


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
    """True if `pattern` proves present in `content`, by either match strategy."""
    if pattern in content:
        return True
    try:
        rx = re.compile(pattern, re.MULTILINE | re.DOTALL)
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

    print(f"fork-patches.txt: {len(rows)} rows total")
    print(f"  {checked} current-assertion rows checked")
    print(f"  {ratchet_skipped} ratchet-log rows skipped (see script docstring, kind 2)")
    print(f"  {historical_skipped} historical/superseded rows skipped (pattern=N/A, kind 3)")
    print()

    ok = True

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

    if ok:
        print(f"OK: all {checked} current-assertion patches are present.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
