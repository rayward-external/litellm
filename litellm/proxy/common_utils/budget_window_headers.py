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

from typing import Any, Dict, List, Optional

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


def format_budget_windows(windows: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    """Render enforcement's per-window snapshot as one header value.

    Returns None when there is nothing to say, which keeps the header absent
    rather than publishing an empty or placeholder value.
    """
    if not windows:
        return None

    items = []
    for window in windows:
        duration = window.get("budget_duration")
        limit = window.get("max_budget")
        spent = window.get("spent")
        # A window missing either half is skipped rather than guessed at. A
        # header that says `limit=None` is worse than one that omits the window:
        # the customer would treat it as a real number.
        if not duration or limit is None or spent is None:
            continue
        # An infinite cap is not a cap. Publishing `limit=inf` invites a client
        # to parse it as a float and get inf, so the window is simply omitted.
        if limit == float("inf"):
            continue
        items.append(f"{duration};limit={_num(limit)};spent={_num(spent)}")

    if not items:
        return None
    return ", ".join(items)


def _num(value: float) -> str:
    """Format a money amount without float noise or scientific notation.

    ``repr(0.1 + 0.2)`` is ``0.30000000000000004``; a customer reading a spend
    figure should not see that. Trailing zeros are trimmed so a whole-dollar cap
    reads ``50`` rather than ``50.000000``.
    """
    formatted = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return formatted or "0"
