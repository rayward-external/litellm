"""RAYWARD FORK PATCH tests — x-usage-budget must actually ship.

The defect these cover is not "the value is wrong". It is that the header was
ABSENT entirely for every key the broker issues, because ``get_custom_headers``
read a lifetime ``max_budget`` that broker keys deliberately never set, and the
resulting ``"None"`` was filtered out before the external rename ran.

So the tests that matter most here are the WIRING ones at the bottom: a correct
formatter that nothing calls reproduces the original bug exactly.
"""

import ast
from pathlib import Path

import pytest

from litellm.proxy.common_utils.budget_window_headers import (
    BUDGET_HEADER_NAME,
    NEUTRAL_BUDGET_HEADER_NAME,
    BudgetWindow,
    format_budget_windows,
)

REPO_ROOT = Path(__file__).resolve().parents[4]


def w(duration, limit, spent):
    return BudgetWindow(budget_duration=duration, max_budget=limit, spent=spent)


class TestFormatBudgetWindows:
    def test_renders_every_window_with_both_halves(self):
        """`limit` AND `spent`, because headroom must come from ONE header.

        x-usage-spend is the key's LIFETIME spend while the caps are windowed, so
        the two cannot be subtracted. Adding a fourth header was declined, so
        this one has to be self-sufficient.
        """
        got = format_budget_windows([w("1d", 50.0, 12.3), w("1w", 200.0, 40.1), w("1mo", 500.0, 95.2)])
        assert got == "1d;limit=50;spent=12.3, 1w;limit=200;spent=40.1, 1mo;limit=500;spent=95.2"

    def test_window_order_is_preserved(self):
        """Enforcement's evaluation order, not a sort. A reorder would be a silent
        change to a published contract."""
        got = format_budget_windows([w("1mo", 500.0, 1.0), w("1d", 50.0, 2.0)])
        assert got.startswith("1mo;")

    @pytest.mark.parametrize("windows", [None, [], [object()]])
    def test_nothing_to_say_means_no_header(self, windows):
        """None keeps the header ABSENT. An empty or placeholder value would read
        as a real cap of zero."""
        assert format_budget_windows(windows) is None

    def test_infinite_cap_is_omitted_not_published(self):
        """`limit=inf` parses to float('inf') in a client and reads as a number."""
        assert format_budget_windows([w("1d", float("inf"), 5.0)]) is None

    def test_a_partial_window_is_skipped_not_guessed(self):
        assert format_budget_windows([w("1d", 50.0, None)]) is None
        assert format_budget_windows([w("1d", None, 5.0)]) is None
        assert format_budget_windows([w(None, 50.0, 5.0)]) is None

    def test_a_partial_window_does_not_suppress_a_good_one(self):
        got = format_budget_windows([w("1d", None, None), w("1w", 200.0, 40.0)])
        assert got == "1w;limit=200;spent=40"

    def test_float_noise_never_reaches_the_customer(self):
        """repr(0.1 + 0.2) is 0.30000000000000004. A spend figure must not look
        like that."""
        assert format_budget_windows([w("1d", 50.0, 0.1 + 0.2)]) == "1d;limit=50;spent=0.3"

    def test_no_scientific_notation(self):
        """A per-request cost of 1.14e-05 is real; a header a customer parses as a
        float must not carry an exponent."""
        got = format_budget_windows([w("1d", 50.0, 0.0000114)])
        assert "e-" not in got and "E-" not in got

    def test_zero_spend_renders_as_zero(self):
        assert format_budget_windows([w("1mo", 500, 0)]) == "1mo;limit=500;spent=0"

    def test_value_is_parseable_back_into_numbers(self):
        """The point of the format: a customer computes headroom from it."""
        raw = format_budget_windows([w("1d", 50.0, 12.3), w("1w", 200.0, 40.1)])
        parsed = {}
        for item in raw.split(", "):
            name, *params = item.split(";")
            fields = dict(p.split("=") for p in params)
            parsed[name] = (float(fields["limit"]), float(fields["spent"]))
        assert parsed == {"1d": (50.0, 12.3), "1w": (200.0, 40.1)}
        assert parsed["1d"][0] - parsed["1d"][1] == pytest.approx(37.7)


class TestWiring:
    """A correct formatter that nothing calls IS the original bug.

    Asserted against the source rather than by booting the proxy: these are
    three separate files a rebase can revert independently, and the failure mode
    is silence, not an exception.
    """

    def test_auth_check_stashes_the_snapshot(self):
        src = (REPO_ROOT / "litellm/proxy/auth/auth_checks.py").read_text()
        assert "valid_token.budget_window_usage" in src, (
            "_virtual_key_multi_budget_check no longer records the per-window snapshot. "
            "It is the only place the numbers exist — get_custom_headers is sync and "
            "cannot await the counters itself."
        )
        assert "spent=window_spend" in src, (
            "the snapshot no longer carries the spend enforcement actually read; "
            "without it the header can only publish limits and headroom is not derivable"
        )
        assert "BudgetWindow(" in src, (
            "the snapshot is no longer built as a BudgetWindow; a dict would trip the "
            "type-discipline gate (LIT001/LIT002) and _render_window reads by attribute"
        )

    def test_header_builder_calls_the_formatter(self):
        """Asserted over the AST, not the source text.

        The previous version matched one exact single-line spelling of the call.
        `ruff format` reflows that call across three lines the moment anything
        near it grows, which reds the guard without a behaviour change and
        teaches whoever hits it to edit the assertion — the failure mode
        described in the repo's own notes on vacuous guards.
        """
        src = (REPO_ROOT / "litellm/proxy/common_request_processing.py").read_text()
        tree = ast.parse(src)

        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "format_budget_windows"
        ]
        # Two: the neutral x-usage-budget for internal callers, and the
        # fork-only branded twin the external rename reads.
        assert len(calls) == 2, (
            f"expected 2 format_budget_windows calls (neutral + branded twin), found "
            f"{len(calls)}; get_custom_headers is no longer publishing both, so one "
            f"audience silently loses its budget header"
        )

        for call in calls:
            assert len(call.args) == 1, "format_budget_windows takes the snapshot positionally"
            arg = call.args[0]
            # getattr, NOT direct attribute access. Direct access raises
            # AttributeError on every request if a sync reverts the _types.py
            # half, and breaks upstream tests that build
            # MagicMock(spec=UserAPIKeyAuth) — pydantic v2 fields are invisible
            # to a mock spec. Both were observed 2026-08-02.
            assert isinstance(arg, ast.Call) and getattr(arg.func, "id", None) == "getattr", (
                "direct attribute access is back; it turns a dropped fork patch into a 500 "
                "on every request and reds upstream's mock-based tests"
            )
            assert len(arg.args) == 3, "getattr without a default still raises when a sync drops the _types.py field"

        assert "from litellm.proxy.common_utils.budget_window_headers import" in src, "the formatter import was dropped"

    def test_snapshot_field_survives_on_the_auth_object(self):
        src = (REPO_ROOT / "litellm/proxy/_types.py").read_text()
        assert "budget_window_usage" in src, (
            "UserAPIKeyAuth.budget_window_usage is gone; the snapshot has nowhere to live"
        )

    def test_the_upstream_header_carries_upstream_value_only(self):
        """`x-litellm-key-max-budget` must be upstream's scalar and nothing else.

        This guard used to assert the OPPOSITE — that the windowed value fell back
        to `or str(user_api_key_dict.max_budget)` on the same header. That design
        put two value types under one upstream name, which no reader can dispatch
        on: the fork's own `gateway_cost.py` did `float(raw)` and reported
        `max_budget: None` for every windowed key, silently, from the day it
        shipped. #491 gave the windows a name of their own.

        Asserted over the AST so the check is about the VALUE bound to that key,
        not about text appearing somewhere in a 2000-line file.
        """
        src = (REPO_ROOT / "litellm/proxy/common_request_processing.py").read_text()
        tree = ast.parse(src)

        values = [
            value
            for node in ast.walk(tree)
            if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and key.value == "x-litellm-key-max-budget"
        ]
        assert len(values) == 1, f"expected exactly one emission of x-litellm-key-max-budget, found {len(values)}"
        assert ast.unparse(values[0]) == "str(user_api_key_dict.max_budget)", (
            f"x-litellm-key-max-budget is not upstream's plain scalar any more, it is "
            f"{ast.unparse(values[0])!r}. Overloading an upstream header with a second "
            f"value type is what #491 removed — put new shapes on a name of our own."
        )

    def test_the_neutral_budget_header_is_emitted_at_source(self):
        """Internal callers get `x-usage-budget` without the external rename.

        The neutral trio is the one documented contract
        (docs/external-api-usage-headers.md). Before #491 it existed only as a
        rename inside the external-audience middleware, so a client written
        against the docs read nothing on `litellm.rayward.ai`. Losing this line
        restores that split silently — internal keeps working for anyone reading
        the branded names, which is exactly who does not notice.
        """
        src = (REPO_ROOT / "litellm/proxy/common_request_processing.py").read_text()
        tree = ast.parse(src)

        emitted = {
            key.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Dict)
            for key in node.keys
            if isinstance(key, ast.Constant) and str(key.value).startswith("x-usage-")
        }
        assert emitted == {"x-usage-cost", "x-usage-spend", NEUTRAL_BUDGET_HEADER_NAME}, (
            f"get_custom_headers emits {sorted(emitted)}; internal callers need all three neutral names at source"
        )

    def test_header_name_matches_the_external_rename_table(self):
        """The external middleware renames exactly one internal name to
        x-usage-budget. If they drift, the windows are computed and then dropped
        by the allowlist — silently, on the external path only."""
        src = (REPO_ROOT / "litellm/proxy/middleware/external_audience_middleware.py").read_text()
        assert f'("{BUDGET_HEADER_NAME}", "x-usage-budget")' in src, (
            f"{BUDGET_HEADER_NAME} is no longer renamed to x-usage-budget; external callers "
            f"would lose the header entirely while internal ones kept it"
        )

    def test_header_builder_emits_under_that_exact_name(self):
        src = (REPO_ROOT / "litellm/proxy/common_request_processing.py").read_text()
        tree = ast.parse(src)
        names = {
            node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and node.value == BUDGET_HEADER_NAME
        }
        assert names, f"get_custom_headers no longer emits {BUDGET_HEADER_NAME}"


class TestNonFiniteLimits:
    """`== float("inf")` is False for BOTH nan and -inf.

    Either would have been formatted straight into the header, and a customer
    parsing float(fields["limit"]) would get nan or -inf and compute nonsense
    headroom — the exact failure filtering +inf exists to prevent.
    """

    @pytest.mark.parametrize("bad", [float("nan"), float("-inf"), float("inf")])
    def test_non_finite_limit_is_omitted(self, bad):
        assert format_budget_windows([w("1d", bad, 5.0)]) is None

    @pytest.mark.parametrize("bad", [float("nan"), float("-inf"), float("inf")])
    def test_non_finite_spend_is_omitted(self, bad):
        assert format_budget_windows([w("1d", 50.0, bad)]) is None

    def test_a_non_finite_window_does_not_suppress_a_good_one(self):
        got = format_budget_windows([w("1d", float("nan"), 5.0), w("1w", 200.0, 40.0)])
        assert got == "1w;limit=200;spent=40"

    def test_no_header_value_can_ever_carry_nan_or_inf(self):
        """Belt and braces: whatever survives must parse as a finite float."""
        import math as _math

        got = format_budget_windows([w("1d", 50.0, 12.3), w("1w", float("nan"), 1.0), w("1mo", float("inf"), 2.0)])
        for item in got.split(", "):
            for param in item.split(";")[1:]:
                assert _math.isfinite(float(param.split("=", 1)[1]))

    def test_a_non_numeric_limit_does_not_raise(self):
        """float("abc") raises ValueError; a header builder must never 500."""
        assert format_budget_windows([w("1d", "abc", 5.0)]) is None
        assert format_budget_windows([w("1d", 50.0, object())]) is None


class TestNeverRaises:
    """This runs while building response headers.

    Anything it raises 500s a request whose completion already SUCCEEDED — the
    user is billed, the tokens are spent, and the caller gets an error. So every
    unexpected input must degrade to "no header", never to an exception.

    Not hypothetical: a plain `Mock` is truthy and not iterable, and the
    comprehension raised `TypeError: 'Mock' object is not iterable`, reddening
    an unrelated upstream pass-through test in CI.
    """

    @pytest.mark.parametrize(
        "junk",
        [
            "1d;limit=50",  # a str is a Sequence — iterating yields chars
            42,
            object(),
            {"budget_duration": "1d"},  # a bare dict, not a BudgetWindow
            [None],
            ["not-a-mapping"],
            [[]],
        ],
    )
    def test_unexpected_input_returns_none_instead_of_raising(self, junk):
        assert format_budget_windows(junk) is None

    def test_a_plain_mock_does_not_raise(self):
        from unittest.mock import Mock

        assert format_budget_windows(Mock()) is None

    def test_a_magicmock_does_not_raise(self):
        from unittest.mock import MagicMock

        assert format_budget_windows(MagicMock()) is None

    def test_the_getattr_seam_is_safe_for_any_mock_shape(self):
        """The exact expression get_custom_headers evaluates."""
        from unittest.mock import MagicMock, Mock

        for mock in (Mock(), MagicMock()):
            assert format_budget_windows(getattr(mock, "budget_window_usage", None)) is None
