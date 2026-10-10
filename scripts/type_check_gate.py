#!/usr/bin/env python3
"""Delta-vs-base per-rule gate for basedpyright.

basedpyright's ``--outputjson`` is reduced to a count of errors per *rule*
(``reportAny``, ``reportArgumentType``, ...) at HEAD and at the merge-base with
the branch this change merges into. A rule fails only when its codebase-wide
total grew past the merge-base count, so a change is blamed for the errors it
adds, never for drift that already sits in the base, and an unrelated PR never
inherits a red from what landed next to it: its count equals its base.

``reportAny`` and ``reportExplicitAny`` are the exception, because Any spreads:
a correct change can surface new ones far from the lines it touched. Their
ceiling is the larger of the merge-base count and a fixed codebase-wide cap in
ANY_CAPS, so a change may add some while the total stays under the cap, and the
total can never pass it. The cap moves only when someone lowers it on main.

Installed packages are part of the measurement: a typed dependency that is
present changes what basedpyright can prove (and therefore which diagnostics
fire) versus when it is absent, so counts from two differently provisioned
venvs are not comparable and their comparison produces phantom breaches no
diff hunk explains. The gate therefore provisions its own environment at
``.venv-typecheck`` (a frozen ``uv sync`` of one canonical dependency-group
set, plus a generated Prisma client) and runs every basedpyright pass from it,
so pre-commit, the CI lint job, and the artifact publisher measure one package
set by construction; re-syncs of an up-to-date env are a near-instant no-op.
The group set is folded into the cache and artifact fingerprint, so counts
recorded under a different set are never matched, only recomputed.

The gate runs basedpyright itself, for both the head and the base pass, with
``NODE_OPTIONS`` raised to the heap this repo needs: basedpyright's node
process OOMs at the ~4 GB default, and when callers had to remember the flag,
every hand-copied pipeline (Makefile, CI, a dev running the recipe by hand)
was one forgotten env line away from an 80-second crash. The base pass is a
second basedpyright run over a detached worktree at the merge-base, under the
same environment so import resolution matches, and scripts/lint_base_counts.py
spares it whenever it can: the per-rule counts are cached under the repo's git
common dir keyed by merge-base commit, ``pyrightconfig.json``, ``uv.lock``,
the Prisma schema, and the dependency-group set, and on a disk-cache miss the
artifact publish-lint-base-counts.yml uploaded for the merge-base is
downloaded through the ``gh`` CLI (``--emit-counts-dir`` is the publisher's
entry point); any fetch failure falls back silently to the local base pass, so
the gate never gets worse than it was without CI.

``--outputjson`` is used rather than text diagnostics because the latter wrap
across lines, leaving the ``(reportRule)`` on a continuation line away from the
``- error:`` marker, so line parsing mis-attributes ~60% of errors -- the JSON
carries an unambiguous ``rule`` field.
"""

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Final, NamedTuple

from lint_base_counts import (
    Checker,
    base_counts_cached,
    emit_counts,
    evaluate,
    head_sha,
    resolve_base_point,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PYRIGHT_CONFIG = REPO_ROOT / "pyrightconfig.json"
UV_LOCK = REPO_ROOT / "uv.lock"

# The one environment every basedpyright pass measures in. The group set is
# the slim one the CI publisher has always installed (not bootstrap's fatter
# --extra proxy env), so published and cached counts stay comparable; changing
# it re-keys every cache and artifact fingerprint, so stale counts can never be
# matched.
TYPECHECK_ENV_DIR = REPO_ROOT / ".venv-typecheck"
TYPECHECK_DEP_GROUPS = ("proxy-dev", "e2e-dev")
PRISMA_GENERATE_SCRIPT = REPO_ROOT / "scripts" / "prisma_generate_if_needed.py"
PRISMA_SCHEMA = REPO_ROOT / "litellm" / "proxy" / "schema.prisma"

# basedpyright's node process needs more than the ~4 GB default heap on this
# repo; appended last so it wins node's last-flag-wins resolution over any
# caller-set value while preserving the caller's other NODE_OPTIONS flags.
NODE_HEAP_OPTION = "--max-old-space-size=8192"

# Bucket for a basedpyright diagnostic with no `rule`. Counted so it's gated.
UNCODED = "<uncoded>"

_HUNK: Final = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


class Violation(NamedTuple):
    file: str
    line: int
    rule: str

ANY_CAPS: Final[Mapping[str, int]] = MappingProxyType({"reportAny": 6150, "reportExplicitAny": 1440})


def _to_relative(raw: str, root: Path) -> str | None:
    path = Path(raw)
    absolute = path if path.is_absolute() else root / path
    try:
        return absolute.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def parse_basedpyright_violations(payload: str, root: Path = REPO_ROOT) -> list[Violation]:
    """In-tree basedpyright errors from `--outputjson`. Warnings and
    information are ignored; only `severity == "error"` is gated. Files
    outside `root` (the venv's site-packages, say) are dropped."""
    try:
        data = json.loads(payload or "{}")
    except json.JSONDecodeError as exc:
        sys.stderr.write(
            f"basedpyright did not emit valid JSON ({exc}); it likely crashed or "
            f"printed text before the JSON. First 500 chars of its output:\n"
            f"{payload[:500]}\n"
        )
        raise SystemExit(1) from exc
    violations: list[Violation] = []
    for diag in data.get("generalDiagnostics", []):
        if diag.get("severity") != "error":
            continue
        rel = _to_relative(diag.get("file", ""), root)
        if rel is None:
            continue
        violations.append(Violation(rel, diag.get("range", {}).get("start", {}).get("line", 0) + 1, diag.get("rule") or UNCODED))
    return violations


def count_basedpyright(payload: str, root: Path = REPO_ROOT) -> dict[str, int]:
    """Count in-tree basedpyright errors per rule from `--outputjson`."""
    return dict(Counter(v.rule for v in parse_basedpyright_violations(payload, root)))


def parse_changed_lines(diff_text: str) -> dict[str, set[int]]:
    changed: dict[str, set[int]] = {}
    path = None
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
        elif path and (match := _HUNK.match(line)):
            start = int(match.group(1))
            count = int(match.group(2)) if match.group(2) is not None else 1
            changed.setdefault(path, set()).update(range(start, start + count))
    return changed


def introduced(violations: Sequence[Violation], changed: Mapping[str, set[int]]) -> list[Violation]:
    return [v for v in violations if v.line in changed.get(v.file, set())]


def _run(cmd: list[str], cwd: Path = REPO_ROOT) -> str:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"{cmd[0]} exited {proc.returncode}")
    return proc.stdout


def node_options_with_heap(base_env: Mapping[str, str]) -> str:
    return f"{base_env.get('NODE_OPTIONS', '')} {NODE_HEAP_OPTION}".strip()


def typecheck_python_version() -> str | None:
    """The interpreter version to build the owned env with, read from
    pyrightconfig's `pythonVersion` so the packages installed for basedpyright
    to see always come from the same version it type-checks against."""
    try:
        config = json.loads(PYRIGHT_CONFIG.read_text())
    except (OSError, ValueError):
        return None
    version: Final = config.get("pythonVersion") if isinstance(config, dict) else None
    return version if isinstance(version, str) else None


def typecheck_env_commands(env_dir: Path = TYPECHECK_ENV_DIR) -> tuple[tuple[str, ...], ...]:
    python_pin: Final = typecheck_python_version()
    sync: Final = (
        "uv",
        "sync",
        "--frozen",
        *(("--python", python_pin) if python_pin else ()),
        *(flag for group in TYPECHECK_DEP_GROUPS for flag in ("--group", group)),
    )
    generate: Final = (str(env_dir / "bin" / "python"), str(PRISMA_GENERATE_SCRIPT))
    return (sync, generate)


def _run_provision_step(cmd: tuple[str, ...], env: Mapping[str, str]) -> int:
    proc = subprocess.run(
        list(cmd), cwd=REPO_ROOT, env=dict(env), capture_output=True, text=True
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
    return proc.returncode


def ensure_typecheck_env(
    env_dir: Path = TYPECHECK_ENV_DIR,
    run: Callable[[tuple[str, ...], Mapping[str, str]], int] = _run_provision_step,
) -> Path:
    """Sync the gate-owned venv (and its generated Prisma client) before a
    measurement pass. Unconditional on purpose: an up-to-date env makes both
    steps near-instant no-ops, and skipping them on a heuristic is how the
    measured environment and the fingerprinted one drift apart."""
    if not env_dir.exists():
        sys.stderr.write(
            f"provisioning {env_dir.name} (first run installs packages and "
            "generates the Prisma client; re-runs are near-instant no-ops)\n"
        )
    env: Final = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(env_dir)}
    for cmd in typecheck_env_commands(env_dir):
        if run(cmd, env) != 0:
            raise SystemExit(
                f"could not provision the type-check environment at {env_dir}: "
                f"`{' '.join(cmd)}` failed"
            )
    return env_dir


def run_basedpyright(cwd: Path = REPO_ROOT, env_dir: Path = TYPECHECK_ENV_DIR) -> str:
    """One basedpyright pass over `cwd` from the gate-owned venv, with the
    raised node heap exported.

    `--pythonpath` pins import resolution to the owned env's interpreter; it is
    the only pin that works, because basedpyright auto-detects a `.venv` in the
    project root and that beats both PATH order and VIRTUAL_ENV, silently
    measuring the caller's fatter venv (whose extra typed packages flip
    diagnostics) whenever the repo has one. Exit 0 (clean) and 1 (errors
    found) are both output-bearing runs; anything else is a crash and fails
    loudly instead of reading as zero errors."""
    bin_dir: Final = env_dir / "bin"
    proc = subprocess.run(
        [
            str(bin_dir / "basedpyright"),
            "--outputjson",
            "--pythonpath",
            str(bin_dir / "python"),
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        env={**os.environ, "NODE_OPTIONS": node_options_with_heap(os.environ)},
    )
    if proc.returncode not in (0, 1):
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"basedpyright exited {proc.returncode}")
    return proc.stdout


@contextlib.contextmanager
def _temp_worktree(ref: str) -> Iterator[Path]:
    parent = Path(tempfile.mkdtemp(prefix="bpr_base_"))
    worktree = parent / "wt"
    try:
        _run(["git", "worktree", "add", "--detach", str(worktree), ref])
        yield worktree
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        shutil.rmtree(parent, ignore_errors=True)


def base_counts(ref: str) -> dict[str, int]:
    """basedpyright error counts per rule for the merge-base tree. The head
    config is copied in so the base is judged by today's rules, and the run uses
    the head environment's basedpyright (on PATH) so imports resolve the same."""
    with _temp_worktree(ref) as worktree:
        shutil.copy(PYRIGHT_CONFIG, worktree / "pyrightconfig.json")
        return count_basedpyright(run_basedpyright(worktree), root=worktree)


def environment_fingerprints(
    dep_groups: tuple[str, ...] = TYPECHECK_DEP_GROUPS,
) -> tuple[str, ...]:
    return (
        *(
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (PYRIGHT_CONFIG, UV_LOCK, PRISMA_SCHEMA)
            if path.exists()
        ),
        "groups:" + ",".join(dep_groups),
    )


def checker_identity(dep_groups: tuple[str, ...] = TYPECHECK_DEP_GROUPS) -> Checker:
    return Checker("basedpyright", environment_fingerprints(dep_groups))


def cmd_check(head: Sequence[Violation], base_ref: str) -> None:
    head_counts: Final = dict(Counter(v.rule for v in head))
    if not head_counts:
        print(
            "FAIL: basedpyright produced no errors. The type checker almost certainly "
            "crashed or emitted nothing; refusing to certify a vacuous run."
        )
        raise SystemExit(1)
    base_point: Final = resolve_base_point(base_ref)
    base: Final = base_counts_cached(checker_identity(), base_point, base_counts)
    if not base:
        print(
            f"FAIL: basedpyright produced no errors for the base tree at {base_point[:12]}, "
            "so every rule would look freshly added. The base pass almost certainly "
            "crashed; refusing to blame this change for it."
        )
        raise SystemExit(1)
    judge(head, head_counts, base, base_point)


def judge(head: Sequence[Violation], head_counts: Mapping[str, int], base: Mapping[str, int], base_point: str) -> None:
    breaches: Final = evaluate(head_counts, base, ANY_CAPS)
    if not breaches:
        print(
            f"OK: every basedpyright rule is within its ceiling "
            f"({sum(head_counts.values())} errors total, base {base_point[:12]})"
        )
        return
    print(f"FAIL: basedpyright errors grew past their ceiling (base {base_point[:12]}):")
    diff: Final = _run(["git", "diff", base_point, "--unified=0", "--no-color", "--", "litellm"])
    new: Final = introduced(head, parse_changed_lines(diff))
    for breach in breaches:
        print(f"  {breach.rule}: total {breach.total} over ceiling {breach.ceiling} (this change added {breach.added})")
        for violation in sorted(v for v in new if v.rule == breach.rule):
            print(f"    {violation.file}:{violation.line}")
    print(
        "Reduce the new errors or remove an equal number elsewhere; the ceiling is the merge-base "
        "count, or the cap in ANY_CAPS (scripts/type_check_gate.py) when that is higher."
    )
    summary: Final = "; ".join(f"{b.rule} {b.total}/{b.ceiling} (+{b.added})" for b in breaches)
    print(f"BREACHED RULES: {summary}")
    raise SystemExit(1)


def main() -> None:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="Comparison ref (default: origin's current default branch)")
    parser.add_argument(
        "--emit-counts-dir",
        type=Path,
        help="Write HEAD's per-rule counts to this directory as a base-counts artifact instead of gating",
    )
    args: Final = parser.parse_args()
    from default_branch import resolve_base_ref
    from gate_slot_lock import held_slot

    if args.emit_counts_dir is not None:
        with held_slot():
            ensure_typecheck_env()
            emit_counts(checker_identity(), count_basedpyright(run_basedpyright()), args.emit_counts_dir, head_sha())
        return
    base_ref: Final = resolve_base_ref(args.base, REPO_ROOT)
    with held_slot():
        ensure_typecheck_env()
        cmd_check(parse_basedpyright_violations(run_basedpyright()), base_ref)


if __name__ == "__main__":
    main()
