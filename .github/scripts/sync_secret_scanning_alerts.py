#!/usr/bin/env python3
"""Apply .github/secret-scanning-triage.toml to GitHub secret-scanning alerts.

The fork's maintenance agent cannot read secret-scanning alerts (its egress
proxy refuses the endpoint and the Actions GITHUB_TOKEN has no permission),
so it judges alerts from this job's log and records each confirmed fixture in
the triage file through a reviewed PR. This job is the only writer:

  * open + listed in the triage file          -> resolve with the entry's
    resolution (used_in_tests | false_positive) and reason;
  * resolved BY THIS SCRIPT + entry removed   -> reopen;
  * open + not listed                         -> printed as SECRET UNTRIAGED;
    the run exits 1.

Safety rails: an entry applies only if the alert's locations include the
entry's `path` (a mistyped alert number never resolves the wrong secret);
an alert whose validity is "active" is never resolved, whatever the file
says; secret values are never requested (hide_secret=true) or printed, since
this repository's Actions logs are public. Detection itself is untouched:
there is no paths-ignore here, by design.

Listing passes the generic / non-provider secret types explicitly: without
`secret_type` the API returns only the default provider patterns (a false
zero for postgres connection strings, private keys, passwords, ...).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import tomllib
from sync_dependabot_alerts import COMMENT_MAX, GitHub

MARKER = "[secret-scanning-triage.toml]"
ALLOWED = ("used_in_tests", "false_positive")
GENERIC_TYPES = (
    "password",
    "http_basic_authentication_header",
    "http_bearer_authentication_header",
    "mongodb_connection_string",
    "mysql_connection_url",
    "postgres_connection_string",
    "openssh_private_key",
    "pgp_private_key",
    "rsa_private_key",
    "ec_private_key",
    "generic_private_key",
)


def load_triage(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    entries: dict[int, dict] = {}
    for e in tomllib.loads(path.read_text()).get("alerts", []):
        n = int(e["number"])
        if e.get("resolution") not in ALLOWED:
            raise SystemExit(f"triage entry #{n}: resolution must be one of {ALLOWED}")
        if not e.get("path") or not e.get("reason"):
            raise SystemExit(f"triage entry #{n}: path and reason are required")
        entries[n] = e
    return entries


def list_alerts(gh: GitHub, state: str) -> list[dict]:
    base = f"secret-scanning/alerts?state={state}&per_page=100&hide_secret=true"
    seen: dict[int, dict] = {}
    for query in (base, f"{base}&secret_type={','.join(GENERIC_TYPES)}"):
        for a in gh.paginate(query):
            seen[a["number"]] = a
    return [seen[n] for n in sorted(seen)]


def locations(gh: GitHub, number: int) -> list[str]:
    out = []
    for loc in gh.paginate(f"secret-scanning/alerts/{number}/locations?per_page=100"):
        d = loc.get("details") or {}
        if d.get("path"):
            out.append(f"{d['path']}:{d.get('start_line', '?')}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--triage", default=".github/secret-scanning-triage.toml")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    args = ap.parse_args()

    token = os.environ.get("GH_TOKEN", "")
    if not token:
        print(
            "::error::GH_TOKEN is empty: set the SECURITY_ALERTS_TOKEN repo secret "
            "(fine-grained token, this repo only, Dependabot alerts + Secret scanning alerts: read and write)."
        )
        return 2
    triage = load_triage(Path(args.triage))
    gh = GitHub(token, args.repo)

    if args.dry_run:
        print("DRY RUN — nothing applied; this run is not an alert inventory")
    resolved = reopened = 0
    untriaged: list[str] = []
    for a in list_alerts(gh, "open"):
        n, kind, validity = a["number"], a.get("secret_type"), a.get("validity") or "unknown"
        locs = locations(gh, n)
        entry = triage.get(n)
        where = ", ".join(locs[:3]) + (f" (+{len(locs) - 3} more)" if len(locs) > 3 else "")
        if entry and validity == "active":
            print(f"::error::#{n} is listed in the triage file but GitHub reports the secret ACTIVE: not resolving it")
            untriaged.append(f"SECRET UNTRIAGED #{n} {kind} {where} validity={validity} (ACTIVE: rotate)")
            continue
        if entry and not any(loc.split(":")[0] == entry["path"] for loc in locs):
            print(f"::warning::#{n} triage path {entry['path']} is not among its locations ({where}): not resolving it")
            entry = None
        if entry:
            comment = f"{MARKER} {' '.join(entry['reason'].split())}"
            comment = comment if len(comment) <= COMMENT_MAX else comment[: COMMENT_MAX - 1] + "…"
            print(f"resolve #{n} {kind} {where} as {entry['resolution']}")
            if not args.dry_run:
                gh.patch(
                    n,
                    {"state": "resolved", "resolution": entry["resolution"], "resolution_comment": comment},
                    kind="secret-scanning",
                )
            resolved += 1
        else:
            untriaged.append(f"SECRET UNTRIAGED #{n} {kind} {where} validity={validity}")

    for a in list_alerts(gh, "resolved"):
        mine = (a.get("resolution_comment") or "").startswith(MARKER)
        if mine and a["number"] not in triage:
            print(f"reopen  #{a['number']} {a.get('secret_type')}: its triage entry was removed")
            if not args.dry_run:
                gh.patch(a["number"], {"state": "open"}, kind="secret-scanning")
            reopened += 1
            n = a["number"]
            locs = locations(gh, n)
            untriaged.append(
                f"SECRET UNTRIAGED #{n} {a.get('secret_type')} {', '.join(locs[:3])} "
                f"validity={a.get('validity') or 'unknown'} (reopened)"
            )

    verb = "would " if args.dry_run else ""
    print(f"\n{verb}resolve {resolved}, {verb}reopen {reopened}, untriaged {len(untriaged)}")
    for line in untriaged:
        print(line)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(f"## Secret-scanning triage sync{' (dry run)' if args.dry_run else ''}\n\n")
            f.write(f"{verb}resolved **{resolved}**, {verb}reopened **{reopened}**, untriaged **{len(untriaged)}**\n\n")
            f.writelines(f"- `{line}`\n" for line in untriaged)
    return 1 if untriaged else 0


if __name__ == "__main__":
    sys.exit(main())
