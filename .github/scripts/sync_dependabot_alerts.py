#!/usr/bin/env python3
"""Mirror osv-scanner.toml triage onto GitHub Dependabot alerts.

osv-scanner.toml is where this fork records a dependency advisory it cannot
fix (no patched release, or the patch needs a newer runtime than a resolution
the lock must keep), each entry with a reason and an expiry. GitHub's
Dependabot alerts never see that file, so every triaged advisory stays OPEN
there forever and buries any genuinely new alert.

For every Dependabot alert this script:
  * open + covered by an unexpired entry  -> dismiss as tolerable_risk, with
    the entry's id, expiry and reason as the comment;
  * dismissed BY THIS SCRIPT + no longer covered (entry removed or expired)
    -> reopen, so the advisory is looked at again;
  * open + not covered -> listed as UNCOVERED; the run exits 1.
Alerts dismissed by a person are never touched.

Coverage follows osv-scanner's own semantics:
  * [[IgnoredVulns]] id matches the alert's GHSA id or any of its aliases.
  * [[PackageOverrides]] with vulnerability.ignore = true. With a `version`,
    the alert is covered only if that version is locked in the alert's
    manifest, falls in the alert's vulnerable range, and NO other locked
    version of the package does -- so an advisory against the other (patched)
    resolution of a split lock stays open. Anything this script cannot parse
    is treated as not covered.

Stdlib only. Needs GH_TOKEN with Dependabot alerts read/write; the Actions
GITHUB_TOKEN cannot read Dependabot alerts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import tomllib

API = "https://api.github.com"
MARKER = "[osv-scanner.toml]"
COMMENT_MAX = 280
OSV_ECOSYSTEM = {"pip": "PyPI", "npm": "npm", "rust": "crates.io", "go": "Go"}


@dataclass(frozen=True)
class Entry:
    label: str
    expiry: dt.date | None
    reason: str

    def active(self, today: dt.date) -> bool:
        return self.expiry is None or today < self.expiry


@dataclass(frozen=True)
class Override:
    ecosystem: str | None
    name: str
    version: str | None
    entry: Entry


def norm(ecosystem: str, name: str) -> str:
    if ecosystem in ("pip", "PyPI"):
        return re.sub(r"[-_.]+", "-", name).lower()
    return name.lower()


def as_date(value: object) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def osv_aliases(vuln_id: str) -> list[str]:
    """GHSA/CVE aliases of an OSV id (e.g. RUSTSEC-*, PYSEC-*) from osv.dev."""
    req = urllib.request.Request(f"https://api.osv.dev/v1/vulns/{vuln_id}")
    req.add_header("User-Agent", "dependabot-alert-sync")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return list(json.loads(resp.read()).get("aliases", []))


def expand_aliases(ignored: dict[str, Entry]) -> None:
    """Dependabot alerts carry only GHSA/CVE ids, so an entry keyed by another
    OSV id (RUSTSEC-*, PYSEC-*, GO-*) must also answer to its aliases. On a
    lookup failure the entry keeps only its own id: its alerts then stay
    UNCOVERED (fail-safe), never wrongly dismissed."""
    for vuln_id, entry in list(ignored.items()):
        if vuln_id.startswith(("GHSA-", "CVE-")):
            continue
        try:
            aliases = osv_aliases(vuln_id)
        except (OSError, ValueError) as err:
            print(f"::warning::could not resolve aliases for {vuln_id}: {err}")
            continue
        for alias in aliases:
            ignored.setdefault(alias, entry)


def load_config(path: Path) -> tuple[dict[str, Entry], list[Override]]:
    data = tomllib.loads(path.read_text())
    ignored: dict[str, Entry] = {}
    for v in data.get("IgnoredVulns", []):
        ignored[v["id"]] = Entry(v["id"], as_date(v.get("ignoreUntil")), v.get("reason", ""))
    overrides: list[Override] = []
    for p in data.get("PackageOverrides", []):
        if not p.get("vulnerability", {}).get("ignore") or "name" not in p:
            continue
        version = p.get("version")
        label = f"{p['name']}=={version}" if version else p["name"]
        overrides.append(
            Override(
                p.get("ecosystem"),
                p["name"],
                version,
                Entry(label, as_date(p.get("effectiveUntil")), p.get("reason", "")),
            )
        )
    return ignored, overrides


def locked_versions(root: Path, manifest: str, ecosystem: str, name: str) -> set[str] | None:
    """Every version of `name` locked in `manifest`; None if unreadable."""
    path = root / manifest
    want = norm(ecosystem, name)
    try:
        if path.name in ("uv.lock", "Cargo.lock"):
            pkgs = tomllib.loads(path.read_text()).get("package", [])
            return {p["version"] for p in pkgs if norm(ecosystem, p.get("name", "")) == want and "version" in p}
        if path.name == "package-lock.json":
            pkgs = json.loads(path.read_text()).get("packages", {})
            return {
                meta["version"]
                for key, meta in pkgs.items()
                if key and norm(ecosystem, key.rsplit("node_modules/", 1)[-1]) == want and "version" in meta
            }
    except (OSError, ValueError, KeyError, tomllib.TOMLDecodeError):
        return None
    return None


def version_key(v: str) -> tuple[int, ...] | None:
    v = v.strip()
    if not re.fullmatch(r"\d+(\.\d+)*", v):
        return None
    parts = [int(x) for x in v.split(".")]
    return tuple(parts + [0] * (6 - len(parts)))


def in_range(version: str, vuln_range: str) -> bool | None:
    """Evaluate GitHub's vulnerable_version_range ('>= 2.6.0, < 13.0.0')."""
    have = version_key(version)
    if have is None:
        return None
    for clause in vuln_range.split(","):
        m = re.fullmatch(r"\s*(<=|>=|<|>|=)\s*(\S+)\s*", clause)
        if not m:
            return None
        bound = version_key(m.group(2))
        if bound is None:
            return None
        op = m.group(1)
        ok = {"<": have < bound, "<=": have <= bound, ">": have > bound, ">=": have >= bound, "=": have == bound}[op]
        if not ok:
            return False
    return True


def coverage(
    alert: dict, ignored: dict[str, Entry], overrides: list[Override], root: Path, today: dt.date
) -> Entry | None:
    adv = alert["security_advisory"]
    ids = {adv["ghsa_id"], *(i["value"] for i in adv.get("identifiers", []))}
    for i in sorted(ids):
        e = ignored.get(i)
        if e and e.active(today):
            return e
    dep = alert["dependency"]
    eco, name = dep["package"]["ecosystem"], dep["package"]["name"]
    vuln_range = alert["security_vulnerability"]["vulnerable_version_range"]
    for o in overrides:
        if not o.entry.active(today) or norm(eco, o.name) != norm(eco, name):
            continue
        if o.ecosystem and o.ecosystem != OSV_ECOSYSTEM.get(eco):
            continue
        if o.version is None:
            return o.entry
        locked = locked_versions(root, dep["manifest_path"], eco, name)
        if not locked or o.version not in locked:
            continue
        hits = {v: in_range(v, vuln_range) for v in locked}
        if None in hits.values():
            continue
        if hits[o.version] and not any(hit for v, hit in hits.items() if v != o.version):
            return o.entry
    return None


def comment_for(e: Entry) -> str:
    until = e.expiry.isoformat() if e.expiry else "no expiry"
    text = f"{MARKER} {e.label} until {until}: {' '.join(e.reason.split())}"
    return text if len(text) <= COMMENT_MAX else text[: COMMENT_MAX - 1] + "…"


class GitHub:
    def __init__(self, token: str, repo: str) -> None:
        self.token, self.repo = token, repo

    def _call(self, method: str, url: str, body: dict | None = None) -> tuple[object, str]:
        req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body else None)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        req.add_header("User-Agent", "dependabot-alert-sync")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read() or b"null"), resp.headers.get("Link", "")
        except urllib.error.HTTPError as err:
            detail = err.read().decode(errors="replace")[:300]
            raise SystemExit(f"GitHub {method} {url} -> HTTP {err.code}: {detail}") from err

    def paginate(self, path: str) -> list[dict]:
        url = f"{API}/repos/{self.repo}/{path}"
        out: list[dict] = []
        while url:
            page, link = self._call("GET", url)
            out.extend(page)  # type: ignore[arg-type]
            m = re.search(r'<([^>]+)>;\s*rel="next"', link)
            url = m.group(1) if m else ""
        return out

    def alerts(self) -> list[dict]:
        return self.paginate("dependabot/alerts?state=open,dismissed&per_page=100")

    def patch(self, number: int, body: dict, kind: str = "dependabot") -> None:
        self._call("PATCH", f"{API}/repos/{self.repo}/{kind}/alerts/{number}", body)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--root", default=".", help="checkout holding the config and lockfiles")
    ap.add_argument("--config", default="osv-scanner.toml")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    args = ap.parse_args()

    token = os.environ.get("GH_TOKEN", "")
    if not token:
        print(
            "::error::GH_TOKEN is empty: set the SECURITY_ALERTS_TOKEN repo secret "
            "(fine-grained token, this repo only, Dependabot alerts + Secret scanning alerts: read and write)."
        )
        return 2
    root = Path(args.root)
    ignored, overrides = load_config(root / args.config)
    expand_aliases(ignored)
    today = dt.datetime.now(dt.timezone.utc).date()
    gh = GitHub(token, args.repo)

    if args.dry_run:
        print("DRY RUN — nothing applied; this run is not an alert inventory")
    dismissed = reopened = 0
    uncovered: list[dict] = []
    for a in sorted(gh.alerts(), key=lambda a: a["number"]):
        num, state = a["number"], a["state"]
        e = coverage(a, ignored, overrides, root, today)
        mine = state == "dismissed" and (a.get("dismissed_comment") or "").startswith(MARKER)
        pkg = f"{a['dependency']['package']['name']} ({a['dependency']['manifest_path']})"
        if state == "open" and e:
            print(f"dismiss #{num} {a['security_advisory']['ghsa_id']} {pkg} <- {e.label}")
            if not args.dry_run:
                gh.patch(
                    num,
                    {"state": "dismissed", "dismissed_reason": "tolerable_risk", "dismissed_comment": comment_for(e)},
                )
            dismissed += 1
        elif mine and not e:
            print(
                f"reopen  #{num} {a['security_advisory']['ghsa_id']} {pkg}: its osv-scanner.toml entry is gone or expired"
            )
            if not args.dry_run:
                gh.patch(num, {"state": "open"})
            reopened += 1
            uncovered.append(a)
        elif state == "open":
            uncovered.append(a)

    verb = "would " if args.dry_run else ""
    print(f"\n{verb}dismiss {dismissed}, {verb}reopen {reopened}, uncovered {len(uncovered)}")
    lines = [
        f"## Dependabot alert sync{' (dry run)' if args.dry_run else ''}",
        "",
        f"{verb}dismissed **{dismissed}**, {verb}reopened **{reopened}**, uncovered **{len(uncovered)}**",
        "",
    ]
    if uncovered:
        lines += ["| # | severity | package | manifest | advisory | first patched |", "|---|---|---|---|---|---|"]
        for a in uncovered:
            v = a["security_vulnerability"]
            fixed = (v.get("first_patched_version") or {}).get("identifier") or "none"
            lines.append(
                f"| {a['number']} | {a['security_advisory']['severity']} | "
                f"{a['dependency']['package']['name']} | {a['dependency']['manifest_path']} | "
                f"{a['security_advisory']['ghsa_id']} | {fixed} |"
            )
            print(
                f"UNCOVERED #{a['number']} {a['security_advisory']['severity']} "
                f"{a['dependency']['package']['name']} {a['dependency']['manifest_path']} "
                f"{a['security_advisory']['ghsa_id']} fix {fixed}"
            )
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("\n".join(lines) + "\n")
    return 1 if uncovered else 0


if __name__ == "__main__":
    sys.exit(main())
