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

Several tests exercise the POST-finalization row shape: CheckBatchCost's
_finalized_file_object rewrites file_object from the provider LiteLLMBatch,
overwriting the client-facing `id` with the provider id/ARN and turning
created_at into an epoch int, preserving only a fixed key set. The renderer must
handle both the pre- and post-finalization shapes. Mirrors the mocking patterns
of test_messages_batches_billing.py.
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


class _ListPrisma:
    """Mock that models KEYSET (seek) pagination — NOT skip/take offset.

    It sorts rows by (created_at desc, unified_object_id desc) and, when the
    WHERE carries the handler's seek predicate
    ((created_at < X) OR (created_at == X AND unified_object_id < Y)), returns
    only rows strictly past that cursor. `on_call(call_no, self)` fires BEFORE
    each fetch so a test can simulate a concurrent insert/delete between chunks.
    """

    def __init__(self, rows, error=None, on_call=None):
        self._rows = list(rows)
        self.error = error
        self.on_call = on_call
        self.calls = []
        self.db = SimpleNamespace(litellm_managedobjecttable=SimpleNamespace(find_many=self._find_many))

    @staticmethod
    def _key(row):
        return (row.created_at, row.unified_object_id)

    @staticmethod
    def _seek_key(where):
        """Extract (created_at, unified_object_id) the seek predicate is past."""
        if not where or "OR" not in where:
            return None
        created_lt = unified_lt = None
        for clause in where["OR"]:
            ca = clause.get("created_at")
            if isinstance(ca, dict) and "lt" in ca:
                created_lt = ca["lt"]
            for sub in clause.get("AND", []):
                uo = sub.get("unified_object_id")
                if isinstance(uo, dict) and "lt" in uo:
                    unified_lt = uo["lt"]
        if created_lt is None or unified_lt is None:
            return None
        return (created_lt, unified_lt)

    async def _find_many(self, where=None, order=None, take=None, **kwargs):
        if self.error is not None:
            raise self.error
        self.calls.append({"where": where, "take": take})
        if self.on_call is not None:
            self.on_call(len(self.calls), self)
        rows = sorted(self._rows, key=self._key, reverse=True)  # created_at desc, uoid desc
        seek = self._seek_key(where)
        if seek is not None:
            # strictly past the cursor == (created_at, uoid) tuple < seek tuple
            rows = [r for r in rows if self._key(r) < seek]
        if take is not None:
            rows = rows[:take]
        return rows


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
    stash_created_at="2026-07-20T00:00:00.000000Z",
    col_created_at=None,
    model="claude-opus-4-6",
    total_records=100,
    include_marker=True,
):
    """A pre-finalization LiteLLM_ManagedObjectTable row.

    stash_created_at -> file_object.created_at (RFC3339 string, what the renderer
    reads). col_created_at -> the row's created_at COLUMN (a datetime, what the
    keyset cursor orders/seeks on); defaults to _BASE_DT."""
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
        status=status,
        created_at=col_created_at if col_created_at is not None else _BASE_DT,
        unified_object_id="u-" + client_id,
    )


def _finalized_row(*, stash, provider_response, model_object_id, status="complete", created_by=None, team_id=None):
    """A row AFTER CheckBatchCost finalization — built with the REAL
    _finalized_file_object so the preserve-list behavior is faithfully tested."""
    from litellm_enterprise.proxy.common_utils.check_batch_cost import CheckBatchCost

    job = SimpleNamespace(file_object=json.dumps(stash))
    response = MagicMock()
    response.model_dump_json.return_value = json.dumps(provider_response)
    return SimpleNamespace(
        file_object=CheckBatchCost._finalized_file_object(job, response),
        model_object_id=model_object_id,
        created_by=created_by,
        team_id=team_id,
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
        assert "created_by" not in batch and "team_id" not in batch

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
    """batch_* rows are the OpenAI-shape /v1/batches flow — NOT message batches
    — even when owned by the caller; they must not appear here."""
    rows = [
        _mine(client_id="batch_01OAI", model_object_id="batch_01OAI", include_marker=False),
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
async def test_before_id_across_chunks(monkeypatch):
    """The sliding buffer must span chunk boundaries: chunk size 3, before_id=007,
    limit=3 -> rows 004,005,006 (built across two chunks before the cursor)."""
    monkeypatch.setattr(mb, "_LIST_CHUNK_SIZE", 3)
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
    hash stashed in litellm_attribution — created_by/team_id are both null."""
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
    team_id column (the coarse ownership signal the poller writes)."""
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
    from the preserved litellm_client_batch_id), so it works with retrieve/{id}."""
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
    same model-invocation-job ARN but no litellm_client_batch_id marker (it was
    not created by this route), so it must NOT surface in the Anthropic listing
    — even though the caller happens to own it. The old classifier (ARN
    substring) would have leaked it."""
    row = _finalized_row(
        # No marker in the stash — this row is NOT ours.
        stash={"litellm_attribution": {"user_api_key": "a" * 64}, "model": "claude-x"},
        provider_response={"id": "batch_openai_dialect_1", "status": "completed", "created_at": 1770000000},
        model_object_id=JOB_ARN,  # a Bedrock model-invocation-job ARN
        status="complete",
        created_by="user-1",
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
    """Create-time write must include the finalization-preserved marker."""

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
    file_object = json.loads(prisma.upsert.await_args.kwargs["data"]["create"]["file_object"])
    assert file_object["litellm_client_batch_id"] == "msgbatch_bedrock_abc_deadbeef"


# ── codex P1 #3: owner filter runs across chunks, not a global scan cap ────────


@pytest.mark.asyncio
async def test_owner_row_found_beyond_first_chunk(monkeypatch):
    """codex P1 #3: a caller's row sitting behind many newer foreign rows must
    still be reachable — the scan pages through the table applying the owner
    filter, rather than capping a global window up front."""
    monkeypatch.setattr(mb, "_LIST_CHUNK_SIZE", 2)
    # 5 newer foreign rows, then the caller's own row (oldest) -> 3rd chunk.
    rows = [
        _foreign(client_id=f"msgbatch_F{i}", model_object_id=f"msgbatch_F{i}", col_created_at=_dt(i)) for i in range(5)
    ]
    rows.append(_mine(client_id="msgbatch_MINE", model_object_id="msgbatch_MINE", col_created_at=_dt(5)))
    prisma = _ListPrisma(rows)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    payload = json.loads((await _call(_FakeRequest(), _auth())).body)
    assert [b["id"] for b in payload["data"]] == ["msgbatch_MINE"]
    assert len(prisma.calls) >= 3  # had to page past the foreign rows


@pytest.mark.asyncio
async def test_after_id_pagination_across_chunks(monkeypatch):
    """after_id stays correct when the cursor and its following page straddle
    chunk boundaries."""
    monkeypatch.setattr(mb, "_LIST_CHUNK_SIZE", 2)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma(_many_mine(6)))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    request = _FakeRequest(query_params={"limit": "2", "after_id": "msgbatch_001"})
    payload = json.loads((await _call(request, _auth())).body)
    assert [b["id"] for b in payload["data"]] == ["msgbatch_002", "msgbatch_003"]
    assert payload["has_more"] is True


@pytest.mark.asyncio
async def test_scan_hard_cap_fails_loud(monkeypatch):
    """Case (c): hitting the scan cap BEFORE the caller's page is satisfied is a
    transient FAILURE, not a false-empty success — a 200 here would tell a caller
    in a large shared ledger 'you have no batches'. It must 503 (and still log)."""
    monkeypatch.setattr(mb, "_LIST_CHUNK_SIZE", 2)
    monkeypatch.setattr(mb, "_LIST_SCAN_HARD_CAP", 4)
    # 10 foreign rows, caller owns none reachable within the 4-row cap.
    rows = [
        _foreign(client_id=f"msgbatch_F{i}", model_object_id=f"msgbatch_F{i}", col_created_at=_dt(i)) for i in range(10)
    ]
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma(rows))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))
    warnings = []
    monkeypatch.setattr(mb.verbose_proxy_logger, "warning", lambda *a, **k: warnings.append(a))

    response = await _call(_FakeRequest(), _auth())
    assert response.status_code == 503
    body = json.loads(response.body)
    assert body["type"] == "error"
    assert body["error"]["type"] == "api_error"
    assert warnings, "cap hit must still be logged"


@pytest.mark.asyncio
async def test_genuine_exhaustion_returns_empty_200(monkeypatch):
    """Case (b): the table is genuinely exhausted below the cap with no owned
    match -> an honest empty 200, NOT the cap-hit 503. The two must not conflate."""
    monkeypatch.setattr(mb, "_LIST_CHUNK_SIZE", 2)
    monkeypatch.setattr(mb, "_LIST_SCAN_HARD_CAP", 100)  # far above the row count
    rows = [
        _foreign(client_id=f"msgbatch_F{i}", model_object_id=f"msgbatch_F{i}", col_created_at=_dt(i)) for i in range(3)
    ]
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _ListPrisma(rows))
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    response = await _call(_FakeRequest(), _auth())
    assert response.status_code == 200
    payload = json.loads(response.body)
    assert payload["data"] == []
    assert payload["has_more"] is False


@pytest.mark.asyncio
async def test_keyset_survives_concurrent_insert(monkeypatch):
    """codex P1: keyset (seek), not offset, chunking. A concurrent INSERT into
    the shared table between chunk fetches must not duplicate a boundary row or
    skip an owned one. With OFFSET (skip=), inserting a newer row shifts the
    window down so the 2nd chunk re-serves the last row of the 1st (duplicate);
    keyset seeks by VALUE, so it can't."""
    monkeypatch.setattr(mb, "_LIST_CHUNK_SIZE", 2)
    # Four owned rows, newest-first A, B, C, D.
    rows = [
        _mine(client_id="msgbatch_A", model_object_id="msgbatch_A", col_created_at=_dt(1)),
        _mine(client_id="msgbatch_B", model_object_id="msgbatch_B", col_created_at=_dt(2)),
        _mine(client_id="msgbatch_C", model_object_id="msgbatch_C", col_created_at=_dt(3)),
        _mine(client_id="msgbatch_D", model_object_id="msgbatch_D", col_created_at=_dt(4)),
    ]

    def _insert_newer_row_before_second_fetch(call_no, prisma):
        if call_no == 2:
            # Another request created a batch NEWER than everything already
            # scanned. Under OFFSET this shifts the window and re-serves B.
            prisma._rows.insert(
                0,
                _mine(client_id="msgbatch_NEW", model_object_id="msgbatch_NEW", col_created_at=_dt(0)),
            )

    prisma = _ListPrisma(rows, on_call=_insert_newer_row_before_second_fetch)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)
    monkeypatch.setattr(mb, "_forward_upstream", AsyncMock(side_effect=AssertionError))

    payload = json.loads((await _call(_FakeRequest(), _auth())).body)
    ids = [b["id"] for b in payload["data"]]
    # Every original owned row exactly once — no duplicated boundary row (B), no
    # skipped row. The row inserted after we passed the top is simply not seen.
    assert ids == ["msgbatch_A", "msgbatch_B", "msgbatch_C", "msgbatch_D"]
    assert len(ids) == len(set(ids)), f"duplicate id in {ids}"


@pytest.mark.asyncio
async def test_keyset_tie_breaker_across_shared_created_at(monkeypatch):
    """Rows sharing a created_at must page by the unified_object_id tie-breaker,
    never straddled/duplicated. Chunk size 1 forces a seek across the tie."""
    monkeypatch.setattr(mb, "_LIST_CHUNK_SIZE", 1)
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
