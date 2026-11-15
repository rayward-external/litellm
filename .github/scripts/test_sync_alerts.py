#!/usr/bin/env python3
"""Regression tests for the security-alert-sync scripts. Stdlib only, no network.

    python3 .github/scripts/test_sync_alerts.py

security-alert-sync.yml runs this before every sync (so a broken script never
writes alert state) and on pull requests that touch the scripts.
"""

from __future__ import annotations

import datetime as dt
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sync_dependabot_alerts as dep  # noqa: E402
import sync_secret_scanning_alerts as sec  # noqa: E402

OSV = """
[[IgnoredVulns]]
id = "GHSA-aaaa-aaaa-aaaa"
ignoreUntil = 2099-01-01
reason = "id entry"

[[IgnoredVulns]]
id = "RUSTSEC-2099-0001"
ignoreUntil = 2099-01-01
reason = "rustsec entry"

[[IgnoredVulns]]
id = "GHSA-expd-expd-expd"
ignoreUntil = 2000-01-01
reason = "expired entry"

[[PackageOverrides]]
name = "pillow"
version = "12.3.0"
ecosystem = "PyPI"
vulnerability.ignore = true
effectiveUntil = 2099-01-01
reason = "python 3.10 side only"
"""

UV_LOCK = """
[[package]]
name = "pillow"
version = "12.3.0"

[[package]]
name = "pillow"
version = "13.0.0"
"""

TODAY = dt.date(2026, 11, 9)


def dep_alert(
    number,
    state="open",
    ghsa="GHSA-zzzz-zzzz-zzzz",
    name="tornado",
    rng="< 1.0",
    comment=None,
):
    return {
        "number": number,
        "state": state,
        "dismissed_comment": comment,
        "security_advisory": {"ghsa_id": ghsa, "identifiers": [], "severity": "high"},
        "security_vulnerability": {
            "vulnerable_version_range": rng,
            "first_patched_version": None,
        },
        "dependency": {
            "package": {"ecosystem": "pip", "name": name},
            "manifest_path": "uv.lock",
        },
    }


def sec_alert(number, validity="unknown", comment=None):
    return {
        "number": number,
        "secret_type": "postgres_connection_string",
        "validity": validity,
        "resolution_comment": comment,
    }


class FakeGitHub:
    """Serves canned alert lists by path prefix and records every PATCH."""

    def __init__(self, dependabot=(), sec_open=(), sec_resolved=(), locations=None):
        self.dependabot, self.sec_open, self.sec_resolved = (
            list(dependabot),
            list(sec_open),
            list(sec_resolved),
        )
        self.locs = locations or {}
        self.patched: list[tuple[int, dict, str]] = []

    def paginate(self, path):
        if path.startswith("secret-scanning/alerts?state=open"):
            return self.sec_open
        if path.startswith("secret-scanning/alerts?state=resolved"):
            return self.sec_resolved
        if path.startswith("secret-scanning/alerts/"):
            n = int(path.split("/")[2])
            return [{"details": {"path": p, "start_line": 1}} for p in self.locs.get(n, ["tests/x.py"])]
        return []

    def alerts(self):
        return self.dependabot

    def patch(self, number, body, kind="dependabot"):
        self.patched.append((number, body, kind))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / ".github").mkdir(parents=True, exist_ok=True)
        (self.root / ".github" / "osv-scanner.toml").write_text(OSV)
        (self.root / "uv.lock").write_text(UV_LOCK)
        self.saved = (
            dep.GitHub,
            sec.GitHub,
            dep.osv_aliases,
            sys.argv[:],
            os.environ.get("GH_TOKEN"),
        )
        dep.osv_aliases = lambda vid: {"RUSTSEC-2099-0001": ["GHSA-rust-rust-rust"]}.get(vid, [])
        os.environ["GH_TOKEN"] = "test"
        # Never append fixture inventories to the real Actions job summary.
        self.saved_summary = os.environ.pop("GITHUB_STEP_SUMMARY", None)

    def tearDown(self):
        dep.GitHub, sec.GitHub, dep.osv_aliases, sys.argv[:], token = self.saved
        if token is None:
            os.environ.pop("GH_TOKEN", None)
        else:
            os.environ["GH_TOKEN"] = token
        if self.saved_summary is not None:
            os.environ["GITHUB_STEP_SUMMARY"] = self.saved_summary
        self.tmp.cleanup()

    def run_dep(self, fake, *extra):
        dep.GitHub = lambda *a: fake
        sys.argv = ["dep", "--repo", "o/r", "--root", str(self.root), *extra]
        with redirect_stdout(io.StringIO()):
            return dep.main()

    def run_sec(self, fake, triage="", *extra):
        sec.GitHub = lambda *a: fake
        path = self.root / "triage.toml"
        path.write_text(triage)
        sys.argv = ["sec", "--repo", "o/r", "--triage", str(path), *extra]
        with redirect_stdout(io.StringIO()):
            return sec.main()


class DependabotCoverage(Base):
    def cover(self, alert, today=TODAY):
        ignored, overrides = dep.load_config(self.root / ".github" / "osv-scanner.toml")
        dep.expand_aliases(ignored)
        return dep.coverage(alert, ignored, overrides, self.root, today)

    def test_split_lock_covers_only_the_old_side(self):
        self.assertIsNotNone(self.cover(dep_alert(1, name="Pillow", rng="< 13.0.0")))
        self.assertIsNone(
            self.cover(dep_alert(1, name="Pillow", rng="< 13.1.0")),
            "reaches the patched 13.0.0",
        )
        self.assertIsNone(self.cover(dep_alert(1, name="pillow", rng=">= 13.0.0")))

    def test_expired_unparseable_and_unknown_stay_open(self):
        self.assertIsNone(self.cover(dep_alert(1, ghsa="GHSA-expd-expd-expd")))
        self.assertIsNone(self.cover(dep_alert(1, name="Pillow", rng="< 13.0.0rc1")))
        self.assertIsNone(self.cover(dep_alert(1, name="Pillow", rng="< 13.0.0"), today=dt.date(2099, 1, 2)))

    def test_id_and_osv_alias_match(self):
        self.assertIsNotNone(self.cover(dep_alert(1, ghsa="GHSA-aaaa-aaaa-aaaa")))
        self.assertIsNotNone(self.cover(dep_alert(1, ghsa="GHSA-rust-rust-rust")))


class DependabotSync(Base):
    def alerts(self):
        return [
            dep_alert(1, ghsa="GHSA-aaaa-aaaa-aaaa"),  # open + covered -> dismiss
            dep_alert(
                2,
                state="dismissed",
                ghsa="GHSA-gone-gone-gone",
                comment=dep.MARKER + " old",
            ),  # ours, uncovered -> reopen
            dep_alert(
                3,
                state="dismissed",
                ghsa="GHSA-gone-gone-gone",
                comment="a person did this",
            ),  # human -> untouched
        ]

    def test_dismisses_reopens_and_fails_on_the_reopened_one(self):
        fake = FakeGitHub(dependabot=self.alerts())
        self.assertEqual(self.run_dep(fake), 1)
        self.assertEqual(
            [(n, b["state"]) for n, b, _ in fake.patched],
            [(1, "dismissed"), (2, "open")],
        )
        self.assertTrue(fake.patched[0][1]["dismissed_comment"].startswith(dep.MARKER))

    def test_dry_run_writes_nothing(self):
        fake = FakeGitHub(dependabot=self.alerts())
        self.run_dep(fake, "--dry-run")
        self.assertEqual(fake.patched, [])


TRIAGE = """
[[alerts]]
number = 10
path = "tests/x.py"
resolution = "used_in_tests"
reason = "fixture"
"""


class SecretSync(Base):
    def test_listed_alert_with_matching_path_is_resolved(self):
        fake = FakeGitHub(sec_open=[sec_alert(10)])
        self.assertEqual(self.run_sec(fake, TRIAGE), 0)
        self.assertEqual(fake.patched[0][1]["state"], "resolved")
        self.assertTrue(fake.patched[0][1]["resolution_comment"].startswith(sec.MARKER))

    def test_wrong_path_is_not_resolved(self):
        fake = FakeGitHub(sec_open=[sec_alert(10)], locations={10: ["src/real.py"]})
        self.assertEqual(self.run_sec(fake, TRIAGE), 1)
        self.assertEqual(fake.patched, [])

    def test_open_active_secret_is_never_resolved(self):
        fake = FakeGitHub(sec_open=[sec_alert(10, validity="active")])
        self.assertEqual(self.run_sec(fake, TRIAGE), 1)
        self.assertEqual(fake.patched, [])

    def test_removed_entry_reopens_and_fails(self):
        fake = FakeGitHub(sec_resolved=[sec_alert(11, comment=sec.MARKER + " fixture")])
        self.assertEqual(self.run_sec(fake, TRIAGE), 1)
        self.assertEqual(fake.patched, [(11, {"state": "open"}, "secret-scanning")])

    def test_resolved_alert_that_turns_active_is_reopened(self):
        fake = FakeGitHub(sec_resolved=[sec_alert(10, validity="active", comment=sec.MARKER + " fixture")])
        self.assertEqual(self.run_sec(fake, TRIAGE), 1, "entry still listed, but the secret is live")
        self.assertEqual(fake.patched, [(10, {"state": "open"}, "secret-scanning")])

    def test_human_resolution_is_never_touched(self):
        fake = FakeGitHub(sec_resolved=[sec_alert(12, validity="active", comment="revoked by a person")])
        self.assertEqual(self.run_sec(fake, TRIAGE), 0)
        self.assertEqual(fake.patched, [])

    def test_dry_run_writes_nothing(self):
        fake = FakeGitHub(sec_open=[sec_alert(10)], sec_resolved=[sec_alert(11, comment=sec.MARKER)])
        self.run_sec(fake, TRIAGE, "--dry-run")
        self.assertEqual(fake.patched, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
