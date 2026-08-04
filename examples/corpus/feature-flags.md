---
title: Feature flags
category: engineering
---

# Feature flags

Flags decouple deploying code from releasing behaviour, which is what makes small,
frequent, low-risk deploys possible.

## Kinds of flag

**Release flags** hide unfinished work. Short lived; removed as soon as the feature
is fully on.

**Operational flags** — kill switches for expensive or risky paths. Long lived by
design, and the first thing to reach for during an incident.

**Permission flags** gate features by plan or by customer. They live as long as the
product does and belong with the entitlement configuration, not in a flag system.

## Creating and rolling out

Create flags in the console or with `meridian flags create <name> --type release`.
Every flag needs an owner and an expected removal date; both are required fields.

Roll out by percentage: 1%, then 10%, 25%, 50%, 100%, holding at each step long
enough to see the dashboards react — an hour is usually the minimum. Bucketing is by
stable user id hash, so a user's experience does not flip between requests. Target
individual organisations first for a friendly-customer beta.

## Writing flagged code

Read the flag once at the top of a request and pass the result down. Re-reading it
deeper in the call stack invites the same request behaving both ways.

Always have a safe default: if the flag service is unreachable the client returns the
default, and that default must be the old behaviour.

## Cleaning up

A stale flag is technical debt with a branch in it. Flags past their removal date are
reported weekly to their owner, and a release flag still open after 90 days is raised
in the engineering forum. Removing a flag means deleting the dead branch too, not
just hardcoding the value.
