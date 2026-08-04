---
title: Backups
category: engineering
---

# Backups and restore

## Schedule

Primary databases take a full snapshot nightly at 02:00 UTC, with write-ahead log
segments shipped continuously. Together these give **point-in-time recovery to any
second within the last 35 days**.

Object storage is versioned and replicated across three availability zones in the
primary region, plus asynchronous replication to a second region with a lag
typically under five minutes.

## Retention

| Backup type      | Kept for  |
|------------------|-----------|
| Nightly snapshot | 35 days   |
| WAL segments     | 35 days   |
| Weekly full      | 6 months  |
| Monthly full     | 2 years   |

Backups are encrypted with a key distinct from the production key, so compromising
the live system does not grant access to its history.

## Restoring to a point in time

1. Raise a restore request naming the target timestamp in UTC and the reason.
2. A second engineer approves — restores are never single-person operations.
3. The snapshot before that timestamp is restored to a **new** cluster and WAL is
   replayed to the exact second.
4. Verify the restored data before any traffic is redirected.

Restores always land on a new cluster. Restoring over a live database would destroy
the evidence of whatever went wrong, and removes the option of changing your mind.

Expect roughly 25 minutes for a 100 GB database, most of it WAL replay. Our RPO is 5
minutes and our RTO is 1 hour for the primary datastore.

## Testing

A restore is exercised automatically every week into an isolated environment, and
row counts and checksums are compared against expectations. A quarterly game day
runs the full procedure by hand, because the automation eventually hides the fact
that no human remembers how to do it. An untested backup is not a backup.
