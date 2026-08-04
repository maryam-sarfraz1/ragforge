---
title: Rate limits
category: platform
---

# Rate limits

Meridian rate limits per API key, not per organisation, so one noisy integration
cannot starve the others.

## Limits by plan

| Plan       | Sustained (req/min) | Burst | Concurrent jobs |
|------------|---------------------|-------|-----------------|
| Free       | 60                  | 100   | 2               |
| Team       | 600                 | 1,000 | 20              |
| Enterprise | 6,000               | 10,000| 200             |

The sustained figure is enforced with a token bucket refilled continuously. The
burst column is the bucket's depth: you may spend it all at once, but the bucket
then refills at the sustained rate, so a burst is not extra monthly capacity.

## What a throttled request looks like

Throttled requests return HTTP `429` with the body code `ERR_4029`. Three headers
accompany every response, throttled or not:

- `X-RateLimit-Limit` — requests permitted in the current window
- `X-RateLimit-Remaining` — requests still available
- `X-RateLimit-Reset` — seconds until the bucket refills

## Handling throttling

Respect the `Retry-After` header, which is always present on a `429`. Back off
exponentially with jitter; a fixed-interval retry from many workers will
resynchronise into exactly the thundering herd that caused the throttle. The
official SDKs do this for you and retry a `429` up to five times by default.

Bulk endpoints are cheaper per record than single-record calls, so batching is
usually a better fix than a higher limit. `POST /v1/records:batch` accepts 1,000
records per call and counts as a single request against your limit.

Rate limits are separate from monthly usage quotas — exceeding a quota is a billing
event, whereas throttling is a momentary traffic control.
