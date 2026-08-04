---
title: Database migrations
category: engineering
---

# Database migrations

Migrations run separately from application deploys, so that code can always be
rolled back independently of schema.

## Expand and contract

Never change a column in one step. Use three:

1. **Expand** — add the new column or table. Nullable, with a default, and ignored
   by the running code.
2. **Migrate** — backfill in batches, and dual-write from the application so old and
   new stay consistent. Ship a release that reads the new column but tolerates the
   old.
3. **Contract** — once no code reads the old column and the backfill is verified,
   drop it. This lands in a separate release, usually a week later.

The gap between expand and contract is what makes a rollback safe: at every point,
both the previous and current releases work against the schema as it stands.

## Adding a column safely

Add it nullable, or with a default that the database can apply without rewriting the
table. On PostgreSQL 11+ a default is a metadata-only change; on older versions it
rewrites the whole table and takes an `ACCESS EXCLUSIVE` lock. Never add a `NOT NULL`
column with no default to a populated table.

Create indexes with `CREATE INDEX CONCURRENTLY`. It is slower and cannot run inside a
transaction, but it does not block writes. A plain `CREATE INDEX` on a large table
will take the service down.

## Locks and timeouts

Every migration sets `lock_timeout` to 3 seconds and `statement_timeout` to 30. A
migration that cannot get its lock quickly must fail and be retried, rather than
queue behind a long read and block every write that arrives after it.

## Backfills

Backfill in batches of at most 10,000 rows with a pause between them, and make the
job resumable — recording progress so a restart continues rather than starting over.
Never backfill in a single transaction.

## Rolling back

Forward-only. A migration that turns out to be wrong is fixed by writing another
migration, not by reversing the last one, because a reversal usually loses the data
written since. Expand/contract means the previous release keeps working while you
write it.
