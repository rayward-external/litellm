"""RAYWARD FORK PATCH — render a key's concurrent budget windows as one header.

WHY THIS EXISTS

The proxy publishes three usage headers, renamed on the external path to
``x-usage-cost`` / ``x-usage-spend`` / ``x-usage-budget`` so an API customer can
see what a request cost and how much of their allowance is left.

Two of the three worked. ``x-usage-budget`` never shipped at all, for a reason
that is invisible until you follow it end to end:

  * ``get_custom_headers`` sources it from ``user_api_key_dict.max_budget`` — a
    single LIFETIME cap.
  * Our key broker deliberately does not set ``max_budget``. It sets
    ``budget_limits``: up to three CONCURRENT windows (1d / 1w / 1mo). A bare
    ``max_budget`` with no ``budget_duration`` would be a lifetime cap, which is
    not the product.
  * So the value is ``None``, ``str(None)`` is ``"None"``, and the
    ``exclude_values = {"", None, "None"}`` filter drops the header before the
    external rename is ever reached.

Measured on router.trueward.ai, 2026-08-02: ``x-usage-cost`` and
``x-usage-spend`` present, ``x-usage-budget`` absent.

AND THE TWO THAT DID SHIP COULD NOT BE COMBINED

``x-usage-spend`` is the key's LIFETIME spend, while every cap we actually
enforce is windowed. Subtracting one from the other is meaningless, so a customer
could not compute headroom even once the budget header existed. That is why this
header carries ``spent`` ALONGSIDE ``limit`` per window rather than only the
limits: headroom has to be derivable from a single header, because adding a
fourth was declined (owner, 2026-08-02) and ``x-usage-spend`` keeps its existing
meaning so nothing a customer already parses breaks.

WHERE THE NUMBERS COME FROM, AND WHY NOT A SECOND LOOKUP

``_virtual_key_multi_budget_check`` already reads every window's counter on every
request — it has to, that is the enforcement. It stashes what it read on the auth
object and this module formats it. A second read would double the cache round
trips per request to recompute numbers the request already had, and could
disagree with the values enforcement actually acted on.

``get_custom_headers`` is sync, so it could not await those counters itself even
if we wanted it to.

FORMAT

An RFC 8941-flavoured list of items with parameters — one item per window::

    x-usage-budget: 1d;limit=50;spent=12.3, 1w;limit=200;spent=40.1

Named parameters rather than positional ``spent/limit`` so the value is
self-describing and a customer cannot silently transpose the two. Windows are
emitted in the order enforcement evaluated them.

IF THIS PATCH IS DROPPED BY A REBASE

``x-usage-budget`` silently stops shipping again and external customers lose the
only view they have of their own cap.
``tests/test_litellm/proxy/common_utils/test_budget_window_headers.py`` fails if
any part of it goes missing, including the call sites in ``auth_checks.py`` and
``common_request_processing.py``.
"""

import math
from collections.abc import Sequence
from typing import NamedTuple


class BudgetWindow(NamedTuple):
    """One enforced budget window and the spend recorded against it.

    A NamedTuple rather than a dict because the repo's type-discipline gate bans
    mutable collections in annotations (LIT001) and mutable construction
    (LIT002) — and it is the better shape anyway: this record is assembled once
    by enforcement and only ever read, so it should not be mutable.

    Construction is the boundary where `budget_limits` entries (dicts from the
    DB, or pydantic model_dump() output) become something with a known shape.
    Everything downstream reads attributes and still proves the types, because
    the VALUES crossing that boundary are not validated by NamedTuple.
    """

    budget_duration: str
    max_budget: float
    spent: float


#: Header the windows are published under. Deliberately the SAME name the
#: lifetime cap already used, rather than a new one:
#:
#:   * the external rename table maps exactly one internal name to
#:     ``x-usage-budget``, and two would collide into a duplicate header;
#:   * a key with windows currently emits NOTHING here (the "None" filter above),
#:     so there is no existing float value for an internal consumer to break on.
#:
#: A key with a lifetime ``max_budget`` still gets the plain float. The two shapes
#: are mutually exclusive per key by construction: the broker sets one or the
#: other, never both.
BUDGET_HEADER_NAME = "x-litellm-key-max-budget"


def format_budget_windows(windows: Sequence[BudgetWindow] | None) -> str | None:
    """Render enforcement's per-window snapshot as one header value.

    Returns None when there is nothing to say, which keeps the header absent
    rather than publishing an empty or placeholder value.

    The annotation says BudgetWindow, but every read below still proves its own
    types. NamedTuple validates nothing at runtime, and the values originate in
    `budget_limits` rows that may be dicts or pydantic model_dump() output, so
    the shape is known here while the VALUES are not.
    """
    # isinstance BEFORE truthiness, and a CONCRETE list/tuple rather than the
    # Sequence ABC. Two things this catches that `if not windows` does not:
    #
    #   * a str is a Sequence, and iterating one yields characters;
    #   * a plain unittest.mock.Mock is TRUTHY and NOT iterable, so the
    #     comprehension below raised `TypeError: 'Mock' object is not iterable`
    #     and reddened an unrelated upstream pass-through test. That was a
    #     genuine defect wearing a test costume: this function runs while
    #     building response headers, so anything it raises 500s a request whose
    #     completion already succeeded.
    #
    # Publishing nothing is always the safe answer here — the header is
    # informational, and its absence is a state callers already handle.
    if not isinstance(windows, (list, tuple)) or not windows:
        return None
    # tuple(generator), not a list comprehension: a list literal/comprehension is
    # mutable construction (LIT002) and nothing here mutates the result.
    rendered = tuple(item for item in (_render_window(w) for w in windows) if item)
    return ", ".join(rendered) if rendered else None


def _render_window(window: object) -> str | None:
    """One window as `<duration>;limit=<n>;spent=<n>`, or None to omit it.

    Reads by getattr rather than attribute access so an entry that is not a
    BudgetWindow — a Mock, a dict, junk — yields None instead of raising. Same
    reasoning as the sequence check above: nothing here may escape as an
    exception, because this runs while building response headers.
    """
    duration = getattr(window, "budget_duration", None)
    limit = _finite(getattr(window, "max_budget", None))
    spent = _finite(getattr(window, "spent", None))

    # A window missing any part is skipped rather than guessed at. A header that
    # says `limit=None` is worse than one that omits the window: the customer
    # would treat it as a real number.
    if not duration or not isinstance(duration, str) or limit is None or spent is None:
        return None

    return f"{duration};limit={_num(limit)};spent={_num(spent)}"


def _finite(value: object) -> float | None:
    """The value as a finite float, or None if it cannot be published.

    The finiteness test is `math.isfinite`, NOT `== float("inf")`. That
    comparison is False for BOTH nan and -inf, so either would have been
    formatted straight into the header, and a customer parsing
    `float(fields["limit"])` would get nan/-inf and compute nonsense headroom —
    the exact failure that filtering +inf exists to prevent.

    Never raises. This runs while building response headers; a budget figure we
    cannot render is a reason to omit one window, never to fail the request.
    """
    if value is None or isinstance(value, bool):
        # bool is a subclass of int, so float(True) is 1.0 — a budget of "True"
        # is not a budget, and publishing 1 would be worse than omitting it.
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _num(value: float) -> str:
    """Format a money amount without float noise or scientific notation.

    ``repr(0.1 + 0.2)`` is ``0.30000000000000004``; a customer reading a spend
    figure should not see that. Trailing zeros are trimmed so a whole-dollar cap
    reads ``50`` rather than ``50.000000``.
    """
    formatted = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return formatted or "0"
