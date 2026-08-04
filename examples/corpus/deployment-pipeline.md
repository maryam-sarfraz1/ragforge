---
title: Deployment
category: engineering
---

# Deployment

## Merging to main

Every merge to `main` triggers the pipeline: lint, unit tests, integration tests
against ephemeral infrastructure, a container build, then an automatic deploy to
staging. Staging soaks for fifteen minutes under synthetic load while error rate and
latency are compared against the previous build.

Production deploys are automatic if staging is clean, but only between 09:00 and
16:00 on Monday to Thursday. Outside that window a build queues until the next one
opens. Nobody should be shipping something at 18:00 on a Friday that they will have
to debug at 22:00.

## Progressive rollout

Production rollout is staged: 5% of traffic for ten minutes, 25% for ten, then 100%.
At each gate the canary's error rate, p99 latency and saturation are compared with
the incumbent. A regression past threshold rolls back automatically and posts to the
team channel.

## Rolling back

`meridian deploy rollback <service>` returns to the previous image, typically within
90 seconds. Rolling back is always the right first move during an incident — restore
service first, investigate afterwards.

Rollback reverts code, not data. A deploy containing a destructive migration cannot
be undone this way, which is why migrations are separated from code changes and
follow the expand/contract sequence.

## Hotfixes

A hotfix branches from the last released tag, not from `main`, so it carries nothing
unreviewed. It still needs a passing pipeline and a second reviewer. Only an on-call
engineer or a service owner may use the emergency window outside deploy hours, and
doing so posts automatically to the engineering channel.
