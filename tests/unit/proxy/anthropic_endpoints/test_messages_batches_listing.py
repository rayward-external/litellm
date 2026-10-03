"""Unit tests for owner-scoped GET /v1/messages/batches (LIST).

The list route used to hard-403 every non-admin key: the upstream forward
enumerates the proxy's SHARED Anthropic workspace, so exposing it to a virtual
key leaks other tenants' batch ids (codex P1, fork PR #131). That left owners
unable to re-identify their own lost batches. These tests pin the fix:

  * admin keys still forward upstream verbatim (byte-identical to before);
  * non-admin keys get an OWNER-SCOPED listing derived from the billing ledger
    (LiteLLM_ManagedObjectTable) — never the upstream enumeration;
  * a caller sees ONLY its own batches (key-hash / user / team ownership), and
    NONE of another tenant's — the leak the admin gate closed stays closed;
  * rendered ids are the CLIENT-facing ids (Bedrock's owner-tagged id included),
    and they SURVIVE CheckBatchCost finalization (which rewrites the row's
    file_object) so they keep working against GET /v1/messages/batches/{id};
  * results_url points at THIS gateway, never api.anthropic.com;
  * no litellm_attribution (or any other tenant's data) leaks into the payload;
  * with no DB the non-admin still 403s (an upstream fallback would reopen the
    leak).

The listing is now a single INDEXED keyset query (owner_key / team_id /
created_by columns), not a chunked full-table scan. owner_key holds the
submitting key hash (written at create, backfilled by migration) and doubles
as the message-batch route discriminator: OpenAI-dialect /v1/batches rows never
set it, so `owner_key IS NOT NULL` excludes them. `_ListPrisma` below is a
faithful in-memory Prisma that interprets the handler's WHERE / order / take
instead of emulating the old scan.

Several tests exercise the POST-finalization row shape: CheckBatchCost's
_finalized_file_object rewrites file_object from the provider LiteLLMBatch,
overwriting the client-facing `id` with the provider id/ARN and turning
created_at into an epoch int, preserving only a fixed key set (never the
owner_key COLUMN, which survives untouched). The renderer must handle both the
pre- and post-finalization shapes. Mirrors the mocking patterns of
test_messages_batches_billing.py.
"""

import datetime
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm.proxy.anthropic_endpoints.messages_batches as mb
from litellm.proxy._types import UserAPIKeyAuth

GATEWAY_BASE = "https://gateway.example.com"
JOB_ARN = "arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/abc123xyz"

# Row-column created_at is a real datetime in Postgres (the keyset cursor
# compares against it directly). Smaller seconds_ago == newer == sorts first.
_BASE_DT = datetime.datetime(2026, 7, 20, tzinfo=datetime.timezone.utc)

_UNSET = object()  # sentinel so a test can force owner_key=None explicitly


def _dt(seconds_ago):
    return _BASE_DT - datetime.timedelta(seconds=seconds_ago)


def _auth(**overrides):
    defaults = dict(
        api_key="a" * 64,  # UserAPIKeyAuth carries the hash
        user_id="user-1",
        team_id="team-1",
        end_user_id="end-user-1",
        key_alias="alias-1",
    )
    defaults.update(overrides)
    return UserAPIKeyAuth(**defaults)


def _admin(**overrides):
    auth = _auth(**overrides)
    auth.user_role = "proxy_admin"
    return auth


class _FakeURL:
    def __init__(self, query=""):
        self.query = query
        self.scheme = "https"
        self.netloc = "gateway.example.com"


class _FakeRequest:
    """Enough of a Starlette Request for the list handler: query_params for the
    owner-scoped path, url.query for the admin upstream-forward path."""

    def __init__(self, query_params=None, query=""):
        self.query_params = query_params or {}
        self.url = _FakeURL(query=query)


def _match(row, where):
    """Interpret the handler's Prisma-style WHERE against an in-memory row.

    Supports the exact operators the listing emits: AND/OR composition,
    scalar equality (file_purpose / team_id / created_by / model_object_id),
    owner_key IS NULL (scalar None) and IS NOT NULL ({"not": None}),
    model_object_id {"endswith"}, and the keyset {"lt"}/{"gt"} comparisons on
    created_at / unified_object_id."""
    if not where:
        return True
    for key, cond in where.items():
        if key == "AND":
            if not all(_match(row, c) for c in cond):
                return False
        elif key == "OR":
            if not any(_match(row, c) for c in cond):
                return False
        elif key == "owner_key":
            val = getattr(row, "owner_key", None)
            if isinstance(cond, dict):
                assert cond.get("not", _UNSET) is None, f"unhandled owner_key cond: {cond}"
                if val is None:  # IS NOT NULL
                    return False
            elif val != cond:  # scalar equality (incl. None == IS NULL)
                return False
        elif key == "model_object_id":
            val = getattr(row, "model_object_id", None)
            if isinstance(cond, dict):
                # only the Bedrock cursor inversion uses a shape filter (endswith)
                assert "endswith" in cond, f"unhandled model_object_id cond: {cond}"
                if not (isinstance(val, str) and val.endswith(cond["endswith"])):
                    return False
            elif val != cond:
                return False
        elif key in ("created_at", "unified_object_id"):
            val = getattr(row, key, None)
            if isinstance(cond, dict):
                for opname, bound in cond.items():
                    if opname == "lt" and not (val < bound):
                        return False
                    elif opname == "gt" and not (val > bound):
                        return False
                    elif opname not in ("lt", "gt"):
                        raise AssertionError(f"unhandled {key} op: {opname}")
            elif val != cond:
                return False
        elif key in ("file_purpose", "team_id", "created_by"):
            default = "batch" if key == "file_purpose" else None
            if getattr(row, key, default) != cond:
                return False
        else:
            raise AssertionError(f"unhandled where key: {key}")
    return True


def _sort(rows, order):
    result = list(rows)
    for spec in reversed(order):  # secondary key first for a stable multi-key sort
        ((col, direction),) = spec.items()
        result.sort(key=lambda r, c=col: getattr(r, c), reverse=(direction == "desc"))
    return result


class _ListPrisma:
    """In-memory Prisma that filters/sorts rows by the handler's actual query
    (WHERE / order / take), covering find_many (the primary + legacy pages) and
    find_first (the cursor id-inversion). `error` makes every call raise."""

    def __init__(self, rows, error=None):
        self._rows = list(rows)
        self.error = error
        self.calls = []
        self.db = SimpleNamespace(
            litellm_managedobjecttable=SimpleNamespace(find_many=self._find_many, find_first=self._find_first)
        )

    async def _find_many(self, where=None, order=None, take=None, **kwargs):
        if self.error is not None:
            raise self.error
        self.calls.append({"where": where, "take": take, "order": order})
        rows = [r for r in self._rows if _match(r, where or {})]
        if order:
            rows = _sort(rows, order)
        if take is not None:
            rows = rows[:take]
        return rows

    async def _find_first(self, where=None, order=None, **kwargs):
        if self.error is not None:
            raise self.error
        self.calls.append({"where": where, "find_first": True})
        rows = [r for r in self._rows if _match(r, where or {})]
        rows = _sort(rows, order or [{"created_at": "desc"}, {"unified_object_id": "desc"}])
        return rows[0] if rows else None


def _stash(
    *,
    client_id,
    attribution,
    model="claude-opus-4-6",
    total_records=100,
    created_at="2026-07-20T00:00:00.000000Z",
    include_marker=True,
):
    """The file_object _record_batch_for_billing writes at create time."""
    obj = {
        "id": client_id,
        "object": "batch",
        "status": "validating",
        "model": model,
        "total_records": total_records,
        "litellm_attribution": attribution,
        "created_at": created_at,
    }
    if include_marker:
        obj["litellm_client_batch_id"] = client_id
    return obj


def _row(
    *,
    client_id,
    model_object_id,
    attribution,
    status="validating",
    created_by=None,
    team_id=None,
    owner_key=_UNSET,
    stash_created_at="2026-07-20T00:00:00.000000Z",
    col_created_at=None,
    model="claude-opus-4-6",
    total_records=100,
    include_marker=True,
):
    """A pre-finalization LiteLLM_ManagedObjectTable row.

    owner_key -> the indexed COLUMN the primary query matches on; defaults to
    the attribution's key hash (what the create path / migration backfill
    write). Pass owner_key=None for an OpenAI-dialect /v1/batches row.
    stash_created_at -> file_object.created_at (RFC3339 string, what the
    renderer reads). col_created_at -> the row's created_at COLUMN (a datetime,
    what the keyset cursor orders/seeks on); defaults to _BASE_DT."""
    if owner_key is _UNSET:
        owner_key = attribution.get("user_api_key")
    return SimpleNamespace(
        file_object=json.dumps(
            _stash(
                client_id=client_id,
                attribution=attribution,
                model=model,
                total_records=total_records,
                created_at=stash_created_at,
                include_marker=include_marker,
            )
        ),
        model_object_id=model_object_id,
        created_by=created_by,
        team_id=team_id,
        owner_key=owner_key,
        status=status,
        created_at=col_created_at if col_created_at is not None else _BASE_DT,
        unified_object_id="u-" + client_id,
    )


def _finalized_row(
    *, stash, provider_response, model_object_id, status="complete", created_by=None, team_id=None, owner_key=_UNSET
):
    """A row AFTER CheckBatchCost finalization — built with the REAL
    _finalized_file_object so the preserve-list behavior is faithfully tested.
    owner_key is a COLUMN, so it survives finalization untouched (defaults to
    the stash's attribution key hash; pass None for an OpenAI-dialect row)."""
    from litellm_enterprise.proxy.common_utils.check_batch_cost import CheckBatchCost

    if owner_key is _UNSET:
        owner_key = (stash.get("litellm_attribution") or {}).get("user_api_key")
    job = SimpleNamespace(file_object=json.dumps(stash))
    response = MagicMock()
    response.model_dump_json.return_value = json.dumps(provider_response)
    return SimpleNamespace(
        file_object=CheckBatchCost._finalized_file_object(job, response),
        model_object_id=model_object_id,
        created_by=created_by,
        team_id=team_id,
        owner_key=owner_key,
        status=status,
        created_at=_BASE_DT,  # row COLUMN is a datetime (keyset ordering)
        unified_object_id="u-" + str(provider_response.get("id")),
    )


def _mine(**overrides):
    """A row owned by the default _auth() caller (matches on its key hash)."""
    defaults = dict(
        client_id="msgbatch_01MINE",
        model_object_id="msgbatch_01MINE",
        attribution={"user_api_key": "a" * 64, "user_api_key_user_id": "user-1"},
    )
    defaults.update(overrides)
    return _row(**defaults)


def _foreign(**overrides):
    """A row owned by a DIFFERENT tenant (different key hash / user / team)."""
    defaults = dict(
        client_id="msgbatch_01FOREIGN",
        model_object_id="msgbatch_01FOREIGN",
        attribution={"user_api_key": "f" * 64, "user_api_key_team_id": "other-team"},
        created_by="other-user",
        team_id="other-team",
    )
    defaults.update(overrides)
    return _row(**defaults)


async def _call(request, auth):
    return await mb.list_message_batches(request, SimpleNamespace(), auth)


# ── (a) non-admin, owned rows -> 200, valid shape, gateway results_url ────────


@pytest.mark.asyncio
async def test_non_admin_lists_only_own_batches(monkeypatch):
    monkeypatch.setenv("PROXY_BASE_URL", GATEWAY_BASE)
    rows = [
        _mine(
            client_id="msgbatch_01ENDED", model_object_id="msgbatch_01ENDED", status="complete", col_created_at=_dt(0)
        ),
        _mine(
            client_id="msgbatch_01OPEN", model_object_id="msgbatch_01OPEN", status="validating", col_created_at=_dt(1)
        ),
        _foreign(col_created_at=_dt(2)),
    ]
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma(rows))
    forward = AsyncMock(side_effect=AssertionError("non-admin must not forward upstream"))
    monkeypatch.setattr(mb, "_forward_upstream", forward)

    response = await _call(_FakeRequest(), _auth())
    assert response.status_code == 200
    payload = json.loads(response.body)

    ids = [b["id"] for b in payload["data"]]
    assert ids == ["msgbatch_01ENDED", "msgbatch_01OPEN"]  # foreign excluded, desc order
    assert payload["has_more"] is False
    assert payload["first_id"] == "msgbatch_01ENDED"
    assert payload["last_id"] == "msgbatch_01OPEN"

    for batch in payload["data"]:
        assert batch["type"] == "message_batch"
        assert batch["processing_status"] in ("in_progress", "ended")
        assert set(batch["request_counts"]) == {"processing", "succeeded", "errored", "canceled", "expired"}
        assert "litellm_attribution" not in batch
        assert "litellm_client_batch_id" not in batch
        assert "created_by" not in batch and "team_id" not in batch and "owner_key" not in batch

    ended = payload["data"][0]
    assert ended["processing_status"] == "ended"
    assert ended["results_url"] == f"{GATEWAY_BASE}/v1/messages/batches/msgbatch_01ENDED/results"
    assert "api.anthropic.com" not in ended["results_url"]

    open_batch = payload["data"][1]
    assert open_batch["processing_status"] == "in_progress"
    assert open_batch["results_url"] is None
    forward.assert_not_awaited()


# ── (b) non-admin sees NONE of another tenant's rows ─────────────────────────


@pytest.mark.asyncio
async def test_non_admin_sees_no_foreign_batches(monkeypatch):
    rows = [_foreign(client_id=f"msgbatch_0{i}FOREIGN", model_object_id=f"msgbatch_0{i}FOREIGN") for i in range(3)]
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma(rows))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    response = await _call(_FakeRequest(), _auth())
    assert response.status_code == 200
    payload = json.loads(response.body)
    assert payload["data"] == []
    assert payload["has_more"] is False
    assert payload["first_id"] is None
    assert payload["last_id"] is None


@pytest.mark.asyncio
async def test_owner_row_found_among_many_foreign(monkeypatch):
    """The owner filter runs in SQL: a caller's single row buried behind many
    newer foreign rows is returned by the one indexed query (the old code had
    to page past them chunk by chunk)."""
    rows = [
        _foreign(client_id=f"msgbatch_F{i}", model_object_id=f"msgbatch_F{i}", col_created_at=_dt(i)) for i in range(20)
    ]
    rows.append(_mine(client_id="msgbatch_MINE", model_object_id="msgbatch_MINE", col_created_at=_dt(99)))
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma(rows))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    payload = json.loads((await _call(_FakeRequest(), _auth())).body)
    assert [b["id"] for b in payload["data"]] == ["msgbatch_MINE"]


# ── (c) admin -> upstream forward unchanged (verbatim query string) ──────────


@pytest.mark.asyncio
async def test_admin_forwards_upstream_with_query(monkeypatch):
    sentinel = SimpleNamespace(status_code=200, body=b"{}")
    forward = AsyncMock(return_value=sentinel)
    monkeypatch.setattr(mb, "_forward_upstream", forward)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma([]))

    request = _FakeRequest(query="limit=5&after_id=msgbatch_01X")
    result = await _call(request, _admin())
    assert result is sentinel
    forward.assert_awaited_once_with(request, "GET", "/v1/messages/batches?limit=5&after_id=msgbatch_01X")


@pytest.mark.asyncio
async def test_admin_forwards_upstream_no_query(monkeypatch):
    sentinel = SimpleNamespace(status_code=200, body=b"{}")
    forward = AsyncMock(return_value=sentinel)
    monkeypatch.setattr(mb, "_forward_upstream", forward)
    request = _FakeRequest()
    await _call(request, _admin())
    # empty query string -> no "?" appended (byte-identical to the old handler)
    forward.assert_awaited_once_with(request, "GET", "/v1/messages/batches")


# ── (d) Bedrock-row rendering: id is the client id WITH the owner tag ────────


@pytest.mark.asyncio
async def test_bedrock_row_renders_owner_tagged_client_id(monkeypatch):
    monkeypatch.setenv("PROXY_BASE_URL", GATEWAY_BASE)
    client_id = "msgbatch_bedrock_abc123xyz_deadbeef"
    row = _mine(client_id=client_id, model_object_id=JOB_ARN, status="complete")
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma([row]))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    payload = json.loads((await _call(_FakeRequest(), _auth())).body)
    assert len(payload["data"]) == 1
    batch = payload["data"][0]
    assert batch["id"] == client_id
    assert batch["processing_status"] == "ended"
    assert batch["results_url"] == f"{GATEWAY_BASE}/v1/messages/batches/{client_id}/results"


@pytest.mark.asyncio
async def test_openai_dialect_batch_rows_excluded(monkeypatch):
    """OpenAI-dialect /v1/batches rows never set owner_key, so `owner_key IS
    NOT NULL` excludes them — even when owned by the caller (same created_by).
    The legacy bridge fetches them (created_by match) but `_client_batch_id`
    filters them out, so they still never appear here."""
    rows = [
        _mine(
            client_id="batch_01OAI",
            model_object_id="batch_01OAI",
            include_marker=False,
            owner_key=None,
            created_by="user-1",
        ),
        _mine(client_id="msgbatch_01OK", model_object_id="msgbatch_01OK"),
    ]
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma(rows))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    payload = json.loads((await _call(_FakeRequest(), _auth())).body)
    assert [b["id"] for b in payload["data"]] == ["msgbatch_01OK"]


# ── (e) no-DB non-admin -> 403 (upstream fallback would reopen the leak) ──────


@pytest.mark.asyncio
async def test_non_admin_no_db_refuses(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    forward = AsyncMock(side_effect=AssertionError("must not forward upstream without a DB"))
    monkeypatch.setattr(mb, "_forward_upstream", forward)

    response = await _call(_FakeRequest(), _auth())
    assert response.status_code == 403
    body = json.loads(response.body)
    assert body["error"]["type"] == "permission_error"
    forward.assert_not_awaited()


@pytest.mark.asyncio
async def test_admin_no_db_still_forwards(monkeypatch):
    """Admins keep the historical shared-workspace behavior even without a DB."""
    sentinel = SimpleNamespace(status_code=200, body=b"{}")
    forward = AsyncMock(return_value=sentinel)
    monkeypatch.setattr(mb, "_forward_upstream", forward)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    await _call(_FakeRequest(), _admin())
    forward.assert_awaited_once()


# ── (f) limit + cursor pagination ────────────────────────────────────────────


def _many_mine(n):
    # newest first (desc): msgbatch_000 is newest (col_created_at=_dt(0)).
    return [
        _mine(
            client_id=f"msgbatch_{i:03d}",
            model_object_id=f"msgbatch_{i:03d}",
            col_created_at=_dt(i),
        )
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_limit_caps_page_and_sets_has_more(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma(_many_mine(5)))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    payload = json.loads((await _call(_FakeRequest(query_params={"limit": "2"}), _auth())).body)
    assert [b["id"] for b in payload["data"]] == ["msgbatch_000", "msgbatch_001"]
    assert payload["has_more"] is True
    assert payload["first_id"] == "msgbatch_000"
    assert payload["last_id"] == "msgbatch_001"


@pytest.mark.asyncio
async def test_after_id_pagination(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma(_many_mine(5)))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    request = _FakeRequest(query_params={"limit": "2", "after_id": "msgbatch_001"})
    payload = json.loads((await _call(request, _auth())).body)
    assert [b["id"] for b in payload["data"]] == ["msgbatch_002", "msgbatch_003"]
    assert payload["has_more"] is True


@pytest.mark.asyncio
async def test_after_id_paginates_across_pages(monkeypatch):
    """Walk the whole list forward one page at a time, seeding each request's
    after_id from the previous page's last_id — the keyset cursor must chain
    cleanly with no gaps or repeats."""
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma(_many_mine(5)))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    seen = []
    after_id = None
    for _ in range(5):
        params = {"limit": "2"}
        if after_id is not None:
            params["after_id"] = after_id
        payload = json.loads((await _call(_FakeRequest(query_params=params), _auth())).body)
        seen.extend(b["id"] for b in payload["data"])
        after_id = payload["last_id"]
        if not payload["has_more"]:
            break
    assert seen == [f"msgbatch_{i:03d}" for i in range(5)]
    assert len(seen) == len(set(seen)), f"duplicate id in {seen}"


@pytest.mark.asyncio
async def test_unknown_after_id_returns_empty(monkeypatch):
    """An unknown forward cursor yields an empty page (best-effort, no error)."""
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma(_many_mine(3)))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    request = _FakeRequest(query_params={"after_id": "msgbatch_does_not_exist"})
    payload = json.loads((await _call(request, _auth())).body)
    assert payload["data"] == []
    assert payload["has_more"] is False


@pytest.mark.asyncio
async def test_limit_capped_at_100_and_unknown_params_ignored(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma(_many_mine(3)))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    request = _FakeRequest(query_params={"limit": "9999", "bogus": "x"})
    response = await _call(request, _auth())
    assert response.status_code == 200
    payload = json.loads(response.body)
    assert len(payload["data"]) == 3
    assert payload["has_more"] is False


# ── before_id: the page immediately NEWER than the cursor (backward paging) ────


@pytest.mark.asyncio
async def test_before_id_returns_page_adjacent_to_cursor(monkeypatch):
    """before_id returns the `limit` rows immediately NEWER than the cursor (the
    previous page), NOT the newest `limit` rows. 100 owned rows, before_id=080,
    limit=20 -> rows 060..079."""
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma(_many_mine(100)))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    request = _FakeRequest(query_params={"limit": "20", "before_id": "msgbatch_080"})
    payload = json.loads((await _call(request, _auth())).body)
    assert [b["id"] for b in payload["data"]] == [f"msgbatch_{i:03d}" for i in range(60, 80)]
    assert payload["has_more"] is True  # rows 000..059 are an even-newer page
    assert payload["first_id"] == "msgbatch_060"
    assert payload["last_id"] == "msgbatch_079"


@pytest.mark.asyncio
async def test_before_id_partial_page_has_no_more(monkeypatch):
    """Fewer than `limit` rows newer than the cursor -> the short page, has_more
    False (there is no even-newer page)."""
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma(_many_mine(10)))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    request = _FakeRequest(query_params={"limit": "20", "before_id": "msgbatch_003"})
    payload = json.loads((await _call(request, _auth())).body)
    assert [b["id"] for b in payload["data"]] == ["msgbatch_000", "msgbatch_001", "msgbatch_002"]
    assert payload["has_more"] is False


@pytest.mark.asyncio
async def test_before_id_multi_row_window(monkeypatch):
    """before_id=007, limit=3 over 10 rows -> the 3 rows immediately newer than
    the cursor (004, 005, 006), desc, with an even-newer page remaining."""
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma(_many_mine(10)))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    request = _FakeRequest(query_params={"limit": "3", "before_id": "msgbatch_007"})
    payload = json.loads((await _call(request, _auth())).body)
    assert [b["id"] for b in payload["data"]] == ["msgbatch_004", "msgbatch_005", "msgbatch_006"]
    assert payload["has_more"] is True


@pytest.mark.asyncio
async def test_before_id_not_found_returns_empty(monkeypatch):
    """An unknown before_id yields an empty page (same convention as after_id)."""
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma(_many_mine(5)))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    request = _FakeRequest(query_params={"before_id": "msgbatch_does_not_exist"})
    payload = json.loads((await _call(request, _auth())).body)
    assert payload["data"] == []
    assert payload["has_more"] is False


@pytest.mark.asyncio
async def test_after_id_and_before_id_together_is_400(monkeypatch):
    """The two cursors are mutually exclusive (Anthropic client error)."""
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma(_many_mine(3)))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    request = _FakeRequest(query_params={"after_id": "msgbatch_000", "before_id": "msgbatch_002"})
    response = await _call(request, _auth())
    assert response.status_code == 400
    assert json.loads(response.body)["error"]["type"] == "invalid_request_error"


# ── (g) key-hash-only ownership (no user_id / team_id) still sees its rows ────


@pytest.mark.asyncio
async def test_key_hash_only_ownership(monkeypatch):
    """A key with no user_id and no team_id owns its batches purely via the key
    hash — now the indexed owner_key COLUMN (backfilled from the attribution
    hash); created_by/team_id are both null."""
    caller = _auth(user_id=None, team_id=None, api_key="k" * 64)
    row = _mine(
        client_id="msgbatch_01KEYONLY",
        model_object_id="msgbatch_01KEYONLY",
        attribution={"user_api_key": "k" * 64},
        created_by=None,
        team_id=None,
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma([row]))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    payload = json.loads((await _call(_FakeRequest(), caller)).body)
    assert [b["id"] for b in payload["data"]] == ["msgbatch_01KEYONLY"]


@pytest.mark.asyncio
async def test_team_ownership_via_row_column(monkeypatch):
    """A teammate whose key hash differs still sees a team batch via the row's
    team_id column."""
    caller = _auth(api_key="z" * 64, user_id="teammate", team_id="team-1")
    row = _mine(
        client_id="msgbatch_01TEAM",
        model_object_id="msgbatch_01TEAM",
        attribution={"user_api_key": "a" * 64},  # a DIFFERENT key hash
        team_id="team-1",
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma([row]))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    payload = json.loads((await _call(_FakeRequest(), caller)).body)
    assert [b["id"] for b in payload["data"]] == ["msgbatch_01TEAM"]


# ── DB read failure fails closed (no leak, no crash) ─────────────────────────


@pytest.mark.asyncio
async def test_db_error_fails_closed(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma([], error=RuntimeError("db down")))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    response = await _call(_FakeRequest(), _auth())
    assert response.status_code >= 500
    body = json.loads(response.body)
    assert body["type"] == "error"


# ── keyset tie-breaker: rows sharing a created_at page by unified_object_id ────


@pytest.mark.asyncio
async def test_keyset_tie_breaker_across_shared_created_at(monkeypatch):
    """Rows sharing a created_at must order by the unified_object_id
    tie-breaker, never straddled/duplicated."""
    same = _dt(0)  # identical created_at for all three
    rows = [
        _mine(client_id="msgbatch_T3", model_object_id="msgbatch_T3", col_created_at=same),
        _mine(client_id="msgbatch_T2", model_object_id="msgbatch_T2", col_created_at=same),
        _mine(client_id="msgbatch_T1", model_object_id="msgbatch_T1", col_created_at=same),
    ]
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma(rows))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    payload = json.loads((await _call(_FakeRequest(), _auth())).body)
    ids = [b["id"] for b in payload["data"]]
    # unified_object_id desc within the tie: u-msgbatch_T3 > T2 > T1.
    assert ids == ["msgbatch_T3", "msgbatch_T2", "msgbatch_T1"]
    assert len(ids) == len(set(ids)), f"duplicate id in {ids}"


@pytest.mark.asyncio
async def test_after_id_tie_breaker_seeks_within_shared_created_at(monkeypatch):
    """after_id on a row sharing its created_at with others must seek strictly
    past it on the unified_object_id tie-breaker, not re-serve the tie group."""
    same = _dt(0)
    rows = [
        _mine(client_id="msgbatch_T3", model_object_id="msgbatch_T3", col_created_at=same),
        _mine(client_id="msgbatch_T2", model_object_id="msgbatch_T2", col_created_at=same),
        _mine(client_id="msgbatch_T1", model_object_id="msgbatch_T1", col_created_at=same),
    ]
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma(rows))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    request = _FakeRequest(query_params={"after_id": "msgbatch_T3"})
    payload = json.loads((await _call(request, _auth())).body)
    # T3 is the newest of the tie (uid u-msgbatch_T3); after it -> T2, T1.
    assert [b["id"] for b in payload["data"]] == ["msgbatch_T2", "msgbatch_T1"]


# ── legacy bridge: pre-migration owner_key IS NULL rows are still surfaced ─────


@pytest.mark.asyncio
async def test_legacy_null_owner_key_row_surfaced_via_column(monkeypatch):
    """A pre-migration row the backfill missed (owner_key NULL) but owned via
    the created_by column must NOT be hidden — the legacy bridge query surfaces
    it and it merges into the page in created_at order."""
    monkeypatch.setenv("PROXY_BASE_URL", GATEWAY_BASE)
    rows = [
        _mine(client_id="msgbatch_NEW", model_object_id="msgbatch_NEW", col_created_at=_dt(0)),
        # legacy: owner_key NULL, attribution key hash absent, owned via created_by
        _row(
            client_id="msgbatch_LEGACY",
            model_object_id="msgbatch_LEGACY",
            attribution={"user_api_key_user_id": "user-1"},
            owner_key=None,
            created_by="user-1",
            col_created_at=_dt(1),
        ),
    ]
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma(rows))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    payload = json.loads((await _call(_FakeRequest(), _auth())).body)
    assert [b["id"] for b in payload["data"]] == ["msgbatch_NEW", "msgbatch_LEGACY"]


@pytest.mark.asyncio
async def test_key_hash_only_null_owner_absent_until_poller_backfill(monkeypatch):
    """New posture (owner-approved simplification): the read path has NO key-hash
    scan. A key-hash-only caller (no user_id/team_id) whose rolling-deploy batch
    still has owner_key NULL is served by retrieve-by-id and is NOT in the listing
    yet — Query A can't match a hash. Once the CheckBatchCost poller backfills
    owner_key (within a poll cycle), the indexed primary query picks it up. This
    test pins both halves of that self-healing gap."""
    caller = _auth(user_id=None, team_id=None, api_key="k" * 64)
    null_row = _row(
        client_id="msgbatch_01OLDPOD",
        model_object_id="msgbatch_01OLDPOD",
        attribution={"user_api_key": "k" * 64},  # owned via the key hash only
        owner_key=None,  # old writer never set the column; poller hasn't healed yet
        created_by=None,
        team_id=None,
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma([null_row]))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))
    before = json.loads((await _call(_FakeRequest(), caller)).body)
    assert before["data"] == []  # not scanned at read time; retrieve-by-id meanwhile

    # After the poller backfills owner_key, the indexed primary query surfaces it.
    healed_row = _row(
        client_id="msgbatch_01OLDPOD",
        model_object_id="msgbatch_01OLDPOD",
        attribution={"user_api_key": "k" * 64},
        owner_key="k" * 64,  # poller filled the NULL from the attribution hash
        created_by=None,
        team_id=None,
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma([healed_row]))
    after = json.loads((await _call(_FakeRequest(), caller)).body)
    assert [b["id"] for b in after["data"]] == ["msgbatch_01OLDPOD"]


# ── codex P1 #1 & #2: client id + classification survive finalization ─────────


def test_finalized_file_object_preserves_client_batch_id():
    """The enterprise preserve-list must carry litellm_client_batch_id through
    finalization — otherwise the owner-tagged Bedrock client id is lost (the
    provider id/ARN overwrites `id`) and the LIST can't return a usable id."""
    from litellm_enterprise.proxy.common_utils.check_batch_cost import CheckBatchCost

    client_id = "msgbatch_bedrock_abc123xyz_deadbeef"
    stash = _stash(client_id=client_id, attribution={"user_api_key": "a" * 64})
    job = SimpleNamespace(file_object=json.dumps(stash))
    response = MagicMock()
    response.model_dump_json.return_value = json.dumps({"id": JOB_ARN, "status": "completed"})
    finalized = json.loads(CheckBatchCost._finalized_file_object(job, response))
    assert finalized["litellm_client_batch_id"] == client_id  # preserved
    assert finalized["id"] == JOB_ARN  # provider id overwrote the stashed id


@pytest.mark.asyncio
async def test_finalized_bedrock_row_keeps_client_id(monkeypatch):
    """codex P1 #1: after finalization the Bedrock row's file_object.id is the
    ARN, but the LIST must still surface the owner-tagged client id (recovered
    from the preserved litellm_client_batch_id), so it works with retrieve/{id}.
    The owner_key COLUMN is untouched by finalization, so the row still matches."""
    monkeypatch.setenv("PROXY_BASE_URL", GATEWAY_BASE)
    client_id = "msgbatch_bedrock_abc123xyz_deadbeef"
    row = _finalized_row(
        stash=_stash(client_id=client_id, attribution={"user_api_key": "a" * 64}),
        provider_response={"id": JOB_ARN, "status": "completed", "created_at": 1770000000},
        model_object_id=JOB_ARN,
        status="complete",
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma([row]))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    payload = json.loads((await _call(_FakeRequest(), _auth())).body)
    assert [b["id"] for b in payload["data"]] == [client_id]
    assert payload["data"][0]["results_url"].endswith(f"/v1/messages/batches/{client_id}/results")


@pytest.mark.asyncio
async def test_finalized_openai_dialect_bedrock_job_excluded(monkeypatch):
    """codex P1 #2: a FINALIZED OpenAI-dialect /v1/batches Bedrock job has the
    same model-invocation-job ARN but owner_key NULL and no
    litellm_client_batch_id marker (it was not created by this route), so it
    must NOT surface in the Anthropic listing — even though the caller happens
    to own it (same created_by). The legacy bridge fetches it on the column
    match, then `_client_batch_id` filters it out."""
    row = _finalized_row(
        # No marker in the stash — this row is NOT ours; OpenAI-dialect => owner_key NULL.
        stash={"litellm_attribution": {"user_api_key": "a" * 64}, "model": "claude-x"},
        provider_response={"id": "batch_openai_dialect_1", "status": "completed", "created_at": 1770000000},
        model_object_id=JOB_ARN,  # a Bedrock model-invocation-job ARN
        status="complete",
        created_by="user-1",
        owner_key=None,
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma([row]))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    payload = json.loads((await _call(_FakeRequest(), _auth())).body)
    assert payload["data"] == []


# ── codex P1 #4: finalized epoch timestamps are preserved & stable ────────────


@pytest.mark.asyncio
async def test_finalized_row_preserves_epoch_timestamps(monkeypatch):
    """codex P1 #4: finalized rows store created_at/expires_at as epoch ints.
    The renderer must convert them, not overwrite created_at with now() (which
    made completed batches list as freshly-created and drift expires_at every
    request)."""
    monkeypatch.setenv("PROXY_BASE_URL", GATEWAY_BASE)
    created_epoch = 1770000000  # 2026-02-02T02:40:00Z
    expires_epoch = created_epoch + 24 * 3600
    client_id = "msgbatch_01FINAL"
    row = _finalized_row(
        stash=_stash(client_id=client_id, attribution={"user_api_key": "a" * 64}),
        provider_response={
            "id": "msgbatch_01FINAL",
            "status": "completed",
            "created_at": created_epoch,
            "expires_at": expires_epoch,
        },
        model_object_id="msgbatch_01FINAL",
        status="complete",
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma([row]))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    first = json.loads((await _call(_FakeRequest(), _auth())).body)["data"][0]
    assert first["created_at"] == mb._to_rfc3339(created_epoch)
    assert first["created_at"].startswith("2026-02-02T")  # the real date, not now()
    assert first["expires_at"] == mb._to_rfc3339(expires_epoch)

    # Stable across requests (the bug was a now()-derived value changing each call).
    second = json.loads((await _call(_FakeRequest(), _auth())).body)["data"][0]
    assert second["created_at"] == first["created_at"]
    assert second["expires_at"] == first["expires_at"]


def test_to_rfc3339_accepts_epoch_and_string_and_rejects_junk():
    assert mb._to_rfc3339("2026-07-20T00:00:00.000000Z") == "2026-07-20T00:00:00.000000Z"
    assert mb._to_rfc3339(1770000000).startswith("2026-02-02T")
    assert mb._to_rfc3339(None) is None
    assert mb._to_rfc3339("") is None
    assert mb._to_rfc3339(True) is None  # bool is not a timestamp


@pytest.mark.asyncio
async def test_record_batch_stashes_client_batch_id(monkeypatch):
    """Create-time write must include the finalization-preserved marker AND the
    indexed owner_key column."""

    class _Prisma:
        def __init__(self):
            self.upsert = AsyncMock()
            self.db = SimpleNamespace(litellm_managedobjecttable=SimpleNamespace(upsert=self.upsert))

    prisma = _Prisma()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)
    await mb._record_batch_for_billing(
        provider_batch_id=JOB_ARN,
        router_model_id="dep-id",
        client_batch_id="msgbatch_bedrock_abc_deadbeef",
        model_name="claude-opus-4-6",
        total_records=100,
        user_api_key_dict=_auth(),
    )
    create = prisma.upsert.await_args.kwargs["data"]["create"]
    file_object = json.loads(create["file_object"])
    assert file_object["litellm_client_batch_id"] == "msgbatch_bedrock_abc_deadbeef"
    assert create["owner_key"] == "a" * 64
