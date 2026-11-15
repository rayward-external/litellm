-- Adds `owner_key` to LiteLLM_ManagedObjectTable so the owner-scoped
-- GET /v1/messages/batches listing becomes a single INDEXED keyset query
-- instead of a chunked full-table scan with a Python owner-filter. The
-- column holds the hash of the submitting virtual key (the same value the
-- create path stashes in file_object.litellm_attribution.user_api_key), so
-- ownership can be matched in SQL. It stays NULL for OpenAI-dialect
-- /v1/batches rows (whose write path never sets it) — that NULL doubles as
-- the message-batch route discriminator (an Anthropic message batch always
-- carries owner_key; an OpenAI-dialect Bedrock job never does).
--
-- The composite indexes match the listing query: filter by owner, sort by
-- created_at DESC (with unified_object_id as the keyset tie-breaker). The
-- created_by index covers the user-branch of the owner match, otherwise
-- unindexed. Index names follow Prisma's auto-generated convention so
-- `prisma migrate diff` against the schema is clean.

ALTER TABLE "LiteLLM_ManagedObjectTable" ADD COLUMN IF NOT EXISTS "owner_key" TEXT;

CREATE INDEX IF NOT EXISTS "LiteLLM_ManagedObjectTable_owner_key_created_at_idx" ON "LiteLLM_ManagedObjectTable" ("owner_key", "created_at" DESC);
CREATE INDEX IF NOT EXISTS "LiteLLM_ManagedObjectTable_created_by_created_at_idx" ON "LiteLLM_ManagedObjectTable" ("created_by", "created_at" DESC);

-- Backfill existing message-batch rows so an owner's pre-migration batches
-- stay visible under the new indexed query. file_object is a Json column
-- storing a JSON-ENCODED STRING (double-encoded) on the write path, so the
-- outer string must be unwrapped (#>> '{}') before extracting the nested
-- attribution key; a row already stored as a JSON object is read directly.
-- Rows whose attribution lacks user_api_key (or are OpenAI-dialect) keep
-- owner_key = NULL — that is the intended discriminator, not a miss.
UPDATE "LiteLLM_ManagedObjectTable"
SET "owner_key" = CASE WHEN jsonb_typeof("file_object") = 'string'
   THEN ("file_object" #>> '{}')::jsonb #>> '{litellm_attribution,user_api_key}'
   ELSE "file_object" #>> '{litellm_attribution,user_api_key}' END
WHERE "file_purpose" = 'batch' AND "owner_key" IS NULL;

-- ROLLING DEPLOY NOTE: an OLD (pre-#335) writer can still create a batch AFTER
-- this backfill runs, leaving owner_key NULL until it expires. The listing code
-- self-heals that window (it scans the bounded NULL slice for such rows — see
-- _fetch_legacy_owner_batches), so NO operator action is required. As optional
-- belt-and-suspenders once every old writer has drained, this same UPDATE can
-- be re-run to clear any residual NULLs; the code fallback is the real fix.
