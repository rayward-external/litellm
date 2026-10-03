"""Guards for the owner_key column that backs the indexed owner-scoped
GET /v1/messages/batches listing:

  * the hand-written migration adds the column + both keyset indexes and
    backfills existing message-batch rows (unwrapping the double-encoded
    file_object JSON);
  * all three in-sync prisma schema copies declare the field and the two
    indexes;
  * the CheckBatchCost poller SELF-HEALS owner_key: it folds the key hash into
    the finalization write when the column is NULL, never overwrites an existing
    value (finalization-survival), and runs an idempotent per-cycle sweep so
    rows an old pod already finalized are drained too.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

# repo root: tests/test_litellm/proxy/anthropic_endpoints/<this file>
_REPO_ROOT = Path(__file__).resolve().parents[4]

_MIGRATION_SQL = (
    _REPO_ROOT
    / "litellm-proxy-extras"
    / "litellm_proxy_extras"
    / "migrations"
    / "20260724000000_add_owner_key_to_managed_object"
    / "migration.sql"
)

_SCHEMA_COPIES = (
    _REPO_ROOT / "schema.prisma",
    _REPO_ROOT / "litellm" / "proxy" / "schema.prisma",
    _REPO_ROOT / "litellm-proxy-extras" / "litellm_proxy_extras" / "schema.prisma",
)


def test_migration_adds_column_and_indexes():
    sql = _MIGRATION_SQL.read_text()
    assert 'ADD COLUMN IF NOT EXISTS "owner_key" TEXT' in sql
    assert (
        'CREATE INDEX IF NOT EXISTS "LiteLLM_ManagedObjectTable_owner_key_created_at_idx" '
        'ON "LiteLLM_ManagedObjectTable" ("owner_key", "created_at" DESC)' in sql
    )
    assert (
        'CREATE INDEX IF NOT EXISTS "LiteLLM_ManagedObjectTable_created_by_created_at_idx" '
        'ON "LiteLLM_ManagedObjectTable" ("created_by", "created_at" DESC)' in sql
    )


def test_migration_backfills_double_encoded_attribution():
    sql = _MIGRATION_SQL.read_text()
    # Backfill only message-batch rows still missing the column.
    assert 'UPDATE "LiteLLM_ManagedObjectTable"' in sql
    assert '"file_purpose" = \'batch\' AND "owner_key" IS NULL' in sql
    # Unwrap the double-encoded JSON string before extracting the nested key.
    assert "jsonb_typeof(\"file_object\") = 'string'" in sql
    assert "(\"file_object\" #>> '{}')::jsonb #>> '{litellm_attribution,user_api_key}'" in sql
    assert "\"file_object\" #>> '{litellm_attribution,user_api_key}'" in sql


@pytest.mark.parametrize("schema_path", _SCHEMA_COPIES, ids=lambda p: str(p.relative_to(_REPO_ROOT)))
def test_all_schema_copies_declare_owner_key(schema_path):
    text = schema_path.read_text()
    # Isolate the LiteLLM_ManagedObjectTable model block so we don't match
    # owner_key/indexes belonging to some other model.
    start = text.index("model LiteLLM_ManagedObjectTable {")
    block = text[start : text.index("}", start)]
    assert "owner_key String?" in block
    assert "@@index([owner_key, created_at(sort: Desc)])" in block
    assert "@@index([created_by, created_at(sort: Desc)])" in block


def _job(*, owner_key, attribution_hash="a" * 64):
    """A managed-object row the poller finalizes. owner_key is the COLUMN value;
    attribution_hash (or None) is the submitting key hash in the file_object stash."""
    attribution = {"user_api_key": attribution_hash} if attribution_hash is not None else {}
    return SimpleNamespace(
        id="job-1",
        owner_key=owner_key,
        file_object=json.dumps({"id": "msgbatch_01X", "litellm_attribution": attribution}),
    )


def _finalize_instance():
    from litellm_enterprise.proxy.common_utils.check_batch_cost import CheckBatchCost

    instance = CheckBatchCost.__new__(CheckBatchCost)
    instance._has_batch_processed_column = True
    update_many = AsyncMock(return_value=1)
    instance.prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_managedobjecttable=SimpleNamespace(update_many=update_many))
    )
    return instance, update_many


# ── poller self-heals owner_key: fold at finalization ────────────────────────


def test_augment_update_backfills_null_owner_key_from_attribution():
    """_augment_update_with_owner_key folds the key hash into the finalization
    write when the column is NULL and the attribution stash carries it."""
    from litellm_enterprise.proxy.common_utils.check_batch_cost import CheckBatchCost

    base = {"status": "completed", "file_object": "{}", "batch_processed": True}
    augmented = CheckBatchCost._augment_update_with_owner_key(_job(owner_key=None), base)
    assert augmented["owner_key"] == "a" * 64
    assert base == {"status": "completed", "file_object": "{}", "batch_processed": True}  # input untouched


def test_augment_update_never_overwrites_existing_owner_key():
    """NEVER-OVERWRITE: a row that already has owner_key keeps it — finalization
    fills a genuine NULL gap, it never changes a real value."""
    from litellm_enterprise.proxy.common_utils.check_batch_cost import CheckBatchCost

    base = {"status": "completed"}
    augmented = CheckBatchCost._augment_update_with_owner_key(
        _job(owner_key="real-existing-hash", attribution_hash="different-hash"), base
    )
    assert "owner_key" not in augmented


def test_augment_update_no_owner_key_without_attribution_hash():
    """A NULL-owner row with no recoverable hash (e.g. OpenAI-dialect) is left
    NULL — the column stays absent from the write."""
    from litellm_enterprise.proxy.common_utils.check_batch_cost import CheckBatchCost

    base = {"status": "completed"}
    augmented = CheckBatchCost._augment_update_with_owner_key(_job(owner_key=None, attribution_hash=None), base)
    assert "owner_key" not in augmented


@pytest.mark.asyncio
async def test_finalize_job_backfills_null_owner_key():
    """End to end: _finalize_job forwards the folded owner_key to the DB when the
    row's column is NULL and the attribution hash is available."""
    instance, update_many = _finalize_instance()
    ok = await instance._finalize_job(_job(owner_key=None), {"status": "completed", "batch_processed": True})
    assert ok is True
    forwarded = update_many.await_args.kwargs["data"]
    assert forwarded["owner_key"] == "a" * 64


@pytest.mark.asyncio
async def test_finalize_job_preserves_existing_owner_key():
    """End to end: _finalize_job never sends owner_key when the row already has
    one — the real value is left untouched (finalization-survival preserved)."""
    instance, update_many = _finalize_instance()
    ok = await instance._finalize_job(_job(owner_key="real-hash"), {"status": "completed", "batch_processed": True})
    assert ok is True
    assert "owner_key" not in update_many.await_args.kwargs["data"]


# ── poller self-heals owner_key: idempotent per-cycle sweep ───────────────────


@pytest.mark.asyncio
async def test_backfill_null_owner_keys_sweep_runs_idempotent_update():
    """The per-cycle sweep drains rows an old pod already finalized (never
    revisited by the finalize path): an idempotent NULL-only UPDATE that derives
    owner_key from the double-encoded attribution and skips rows with no hash."""
    from litellm_enterprise.proxy.common_utils.check_batch_cost import CheckBatchCost

    instance = CheckBatchCost.__new__(CheckBatchCost)
    instance._has_batch_processed_column = True
    execute_raw = AsyncMock(return_value=0)
    instance.prisma_client = SimpleNamespace(db=SimpleNamespace(execute_raw=execute_raw))

    await instance._backfill_null_owner_keys()
    sql = execute_raw.await_args.args[0]
    assert 'UPDATE "LiteLLM_ManagedObjectTable"' in sql
    assert '"owner_key"' in sql
    assert '"file_purpose" = \'batch\' AND "owner_key" IS NULL' in sql
    assert "'{litellm_attribution,user_api_key}'" in sql
    assert "IS NOT NULL" in sql  # skip rows with no derivable hash (idempotent)


@pytest.mark.asyncio
async def test_backfill_null_owner_keys_sweep_skipped_on_legacy_schema():
    """A schema without batch_processed predates owner_key, so the sweep is a
    no-op there (never issues the UPDATE that would error on a missing column)."""
    from litellm_enterprise.proxy.common_utils.check_batch_cost import CheckBatchCost

    instance = CheckBatchCost.__new__(CheckBatchCost)
    instance._has_batch_processed_column = False
    execute_raw = AsyncMock()
    instance.prisma_client = SimpleNamespace(db=SimpleNamespace(execute_raw=execute_raw))

    await instance._backfill_null_owner_keys()
    execute_raw.assert_not_awaited()
